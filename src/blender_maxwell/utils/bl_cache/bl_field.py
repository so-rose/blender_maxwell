# blender_maxwell
# Copyright (C) 2024 blender_maxwell Project Contributors
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""Implements various key caches on instances of Blender objects, especially nodes and sockets."""

import functools
import inspect
import typing as typ

import bpy

from blender_maxwell import contracts as ct
from blender_maxwell.utils import bl_instance, logger

from .bl_prop import BLProp
from .bl_prop_type import BLPropType
from .signal import Signal

log = logger.get(__name__)


StringPropSubType: typ.TypeAlias = typ.Literal[
	'FILE_PATH', 'DIR_PATH', 'FILE_NAME', 'BYTE_STRING', 'PASSWORD', 'NONE'
]

StrMethod: typ.TypeAlias = typ.Callable[
	[bl_instance.BLInstance, bpy.types.Context, str], list[tuple[str, str]]
]
EnumMethod: typ.TypeAlias = typ.Callable[
	[bl_instance.BLInstance, bpy.types.Context], list[ct.BLEnumElement]
]

DEFAULT_ENUM_ITEMS_SINGLE = [('NONE', 'None', 'No items...', '', 0)]
DEFAULT_ENUM_ITEMS_MANY = [('NONE', 'None', 'No items...', '', 2**0)]


@functools.cache
def default_enum_items(enum_many: bool) -> list[ct.BLEnumElement]:
	return DEFAULT_ENUM_ITEMS_MANY if enum_many else DEFAULT_ENUM_ITEMS_SINGLE


####################
# - BLField
####################
class BLField:
	"""A descriptor that allows persisting arbitrary types in Blender objects, with cached reads."""

	def __init__(
		self,
		default_value: typ.Any = None,
		use_prop_update: bool = True,
		## Static
		prop_ui: bool = False,  ## TODO: Remove
		abs_min: int | float | None = None,
		abs_max: int | float | None = None,
		soft_min: int | float | None = None,
		soft_max: int | float | None = None,
		float_step: int | None = None,
		float_prec: int | None = None,
		str_secret: bool | None = None,
		path_type: typ.Literal['dir', 'file'] | None = None,
		# blptr_type: typ.Any | None = None,  ## A Blender ID type
		## TODO: Test/Implement
		## Dynamic
		str_cb: StrMethod | None = None,
		enum_cb: EnumMethod | None = None,
		cb_depends_on: set[str] | None = None,
	) -> typ.Self:
		"""Initializes and sets the attribute to a given default value.

		The attribute **must** declare a type annotation, and it **must** match the type of `default_value`.

		Parameters:
			default_value: The default value to use if the value is read before it's set.
			use_prop_update: If True, `BLField` will consent to `bl_instance.on_prop_changed(attr_name)` being run whenever the field is changed.
				UI changes done to the property via Blender **always** trigger `bl_instance.on_bl_prop_changed`; however, the `BLField` decides whether `on_prop_changed` should be run as well.
				That control is offered through `use_prop_update`.
			abs_min: Sets the absolute minimum value of the property.
				Only meaningful for numerical properties.
			abs_max: Sets the absolute maximum value of the property.
				Only meaningful for numerical properties.
			soft_min: Sets a value which will feel like a minimum in the UI, but which can be overridden by setting a value directly.
				In practice, "scrolling" through values will stop here.
				Only meaningful for numerical properties.
			soft_max: Sets a value which will feel like a maximum in the UI, but which can be overridden by setting a value directly.
				In practice, "scrolling" through values will stop here.
				Only meaningful for numerical properties.
			float_step: Sets the interval (/100) of each step when "scrolling" through the values of a float property, aka. the speed.
				Only meaningful for float-like properties.
			float_step: Sets the decimal places of precision to display.
				Only meaningful for float-like properties.
			str_secret: Marks the string as "secret", which prevents its save-persistance, and causes the UI to display dots instead of characters.
				**DO NOT** rely on this property for "real" security.
				_If in doubt, this isn't good enough._
				Only meaningful for `str` properties.
			path_type: Makes the path as pointing to a folder or to a file.
				Only meaningful for `pathlib.Path` properties.
				**NOTE**: No effort is made to make paths portable between operating systems.
				Use with care.
			str_cb: Method used to determine all valid strings, which presents to the user as a fuzzy-style search dropdown.
				Only meaningful for `str` properties.
				Results are not persisted, and must therefore re-run when reloading the file.
				Otherwise, it is cached, but is re-run whenever `Signal.ResetStrSearch` is set.
			enum_cb: Method used to determine all valid enum elements, which presents to the user as a dropdown.
				The caveats with dynamic `bpy.props.EnumProperty`s are **absurdly sharp**.
				Those caveats are entirely mitigated when using this callback, at the cost of manual resets.
				Is re-run whenever `Signal.ResetEnumItems` is set, and otherwise cached both persistently and non-persistently.
			cb_depends_on: Declares that `str_cb` / `enum_cb` should be regenerated whenever any of the given property names change.
				This allows fully automating the invocation of `Signal.ResetEnumItems` / `Signal.ResetStrSearch` in common cases.
		"""
		log.debug(
			'Initializing BLField (default_value=%s, use_prop_update=%s)',
			str(default_value),
			str(use_prop_update),
		)

		self.use_dynamic_enum = enum_cb is not None
		self.use_str_search = str_cb is not None

		## TODO: Use prop_flags
		self.prop_info = {
			'default': default_value,
			'use_prop_update': use_prop_update,
			# Int* | Float*: Bounds
			'min': abs_min,
			'max': abs_max,
			'soft_min': soft_min,
			'soft_max': soft_max,
			# Float*: UI
			'step': float_step,
			'precision': float_prec,
			# BLPointer: ID Type
			#'blptr_type': blptr_type,
			# Str | Path | Enum: Flag Setters
			'str_secret': str_secret,
			'path_type': path_type,
			# Search: Str
			'str_search': self.use_str_search,
			'safe_str_cb': lambda _self, context, edit_text: self.safe_str_cb(
				_self, context, edit_text
			),
			# Search: Enum
			'enum_dynamic': self.use_dynamic_enum,
			'safe_enum_cb': lambda _self, context: self.safe_enum_cb(_self, context),
		}

		# BLProp
		self.bl_prop: BLProp | None = None
		self.bl_prop_enum_items: BLProp | None = None
		self.bl_prop_str_search: BLProp | None = None

		self.enum_cb = enum_cb
		self.str_cb = str_cb

		self.cb_depends_on: set[str] | None = cb_depends_on

		# Update Suppressing
		self.suppress_update: dict[str, bool] = {}

	####################
	# - Descriptor Setup
	####################
	def __set_name__(self, owner: type[bl_instance.BLInstance], name: str) -> None:
		"""Sets up this descriptor on the class, preparing it for per-instance use.

		A `BLProp` is constructed using this descriptor's attribute name on `owner`, and the `self.prop_info` previously created during `self.__init__()`.
		Then, a corresponding / underlying `bpy.types.Property` is initialized on `owner` using `self.bl_prop.init_bl_type(owner)`

		Notes:
			Run by Python when setting an instance of a "descriptor" class, to an attribute of another class (denoted `owner`).
			For more, search for "Python descriptor protocol".

		Parameters:
			owner: The class that contains an attribute assigned to an instance of this descriptor.
			name: The name of the attribute that an instance of descriptor was assigned to.
		"""
		prop_type = inspect.get_annotations(owner).get(name)
		self.bl_prop = BLProp(
			name=name,
			prop_info=self.prop_info,
			prop_type=prop_type,
			bl_prop_type=BLPropType.from_type(prop_type),
		)

		# Initialize Field on BLClass
		self.bl_prop.init_bl_type(
			owner,
			enum_depends_on=self.cb_depends_on,
			strsearch_depends_on=self.cb_depends_on,
		)

		# Dynamic Enum: Initialize Persistent Enum Items
		if self.prop_info['enum_dynamic']:
			self.bl_prop_enum_items = BLProp(
				name=self.bl_prop.enum_cache_key,
				prop_info={'default': [], 'use_prop_update': False},
				prop_type=list[ct.BLEnumElement],
				bl_prop_type=BLPropType.Serialized,
			)
			self.bl_prop_enum_items.init_bl_type(owner)

		# Searched Str: Initialize Persistent Str List
		if self.prop_info['str_search']:
			self.bl_prop_str_search = BLProp(
				name=self.bl_prop.str_cache_key,
				prop_info={'default': [], 'use_prop_update': False},
				prop_type=list[str],
				bl_prop_type=BLPropType.Serialized,
			)
			self.bl_prop_str_search.init_bl_type(owner)

	def __get__(
		self,
		bl_instance: bl_instance.BLInstance | None,
		owner: type[bl_instance.BLInstance],
	) -> typ.Any:
		"""Retrieves the value described by the BLField.

		Notes:
			Run by Python when the attribute described by the descriptor is accessed.
			For more, search for "Python descriptor protocol".

		Parameters:
			bl_instance: Instance that is accessing the attribute.
			owner: The class that owns the instance.
		"""
		# Compute Value (if available)
		cached_value = self.bl_prop.read_nonpersist(bl_instance)
		if cached_value is Signal.CacheNotReady or cached_value is Signal.CacheEmpty:
			if bl_instance is not None:
				persisted_value = self.bl_prop.read(bl_instance)
				self.bl_prop.write_nonpersist(bl_instance, persisted_value)
				return persisted_value
			return self.bl_prop.default_value  ## TODO: Good idea?
		return cached_value

	def suppress_next_update(self, bl_instance) -> None:
		self.suppress_update[bl_instance.instance_id] = True
		## TODO: Make it a context manager to prevent the worst of surprises

	def __set__(
		self, bl_instance: bl_instance.BLInstance | None, value: typ.Any
	) -> None:
		"""Sets the value described by the BLField.

		In general, any BLField modified in the UI will set `InvalidateCache` on this descriptor.
		If `self.prop_info['use_prop_update']` is set, the method `bl_instance.on_prop_changed(self.bl_prop.name)` will then be called and start a `FlowKind.DataChanged` event chain.

		Notes:
			Run by Python when the attribute described by the descriptor is set.
			For more, search for "Python descriptor protocol".

		Parameters:
			bl_instance: Instance that is accessing the attribute.
			owner: The class that owns the instance.
		"""
		# Perform Update Chain
		## -> We still respect 'use_prop_update', since it is user-sourced.
		if value is Signal.DoUpdate:
			if self.prop_info['use_prop_update']:
				bl_instance.on_prop_changed(self.bl_prop.name)

		# Invalidate Cache
		## -> This empties the non-persistent cache.
		## -> As a result, the value must be reloaded from the property.
		## The 'on_prop_changed' method on the bl_instance might also be called.
		elif value is Signal.InvalidateCache or value is Signal.InvalidateCacheNoUpdate:
			self.bl_prop.invalidate_nonpersist(bl_instance)

			# Update Suppression
			if self.suppress_update.get(bl_instance.instance_id):
				self.suppress_update[bl_instance.instance_id] = False

			# ELSE: Trigger Update Chain
			elif self.prop_info['use_prop_update'] and value is Signal.InvalidateCache:
				bl_instance.on_prop_changed(self.bl_prop.name)

		# Reset Enum Items
		elif value is Signal.ResetEnumItems:
			# Retrieve Old Items
			## -> This is verbatim what is being persisted, currently.
			## -> len(0): Manually replaced w/fallback to guarantee >=len(1)
			## -> Fallback element is 'NONE'.
			_old_items: list[ct.BLEnumElement] = self.bl_prop_enum_items.read(
				bl_instance
			)
			old_items = (
				_old_items
				if _old_items
				else default_enum_items(self.bl_prop.is_enum_many)
			)

			# Retrieve Current Items
			## -> len(0): Manually replaced w/fallback to guarantee >=len(1)
			## -> Manually replaced fallback element is 'NONE'.
			_current_items: list[ct.BLEnumElement] = self.enum_cb(bl_instance, None)
			current_items = (
				_current_items
				if _current_items
				else default_enum_items(self.bl_prop.is_enum_many)
			)

			# Compare Old | Current
			## -> We don't involve non-persistent caches (they lie!)
			## -> Since we persist the user callback directly, we can compare.
			if old_items != current_items:
				# Retrieve Old Enum Item
				## -> This is verbatim what is being used.
				## -> De-Coerce None -> 'NONE' to avoid special-cased search.
				_old_item = self.bl_prop.read(bl_instance)
				old_item = 'NONE' if _old_item is None else _old_item

				# Swap Enum Items
				## -> This is the hot stuff - the enum elements are overwritten.
				## -> The safe_enum_cb will pick up on this immediately.
				self.suppress_next_update(bl_instance)
				self.bl_prop_enum_items.write(bl_instance, current_items)

				# Old Item in Current Items
				## -> It's possible that the old enum key is in the new enum.
				## -> If so, the user will expect it to "remain".
				## -> Thus, we set it - Blender sees a change, user doesn't.
				## -> DO NOT trigger on_prop_changed (since "nothing changed").
				if any(old_item == item[0] for item in current_items):
					self.suppress_next_update(bl_instance)
					self.bl_prop.write(bl_instance, old_item)
					## -> TODO: Don't write if not needed.

				# Old Item Not in Current Items
				## -> In this case, fallback to the first current item.
				## -> DO trigger on_prop_changed (since it changed!)
				else:
					_first_current_item = current_items[0][0]
					first_current_item = (
						_first_current_item if _first_current_item != 'NONE' else None
					)

					self.suppress_next_update(bl_instance)
					self.bl_prop.write(bl_instance, first_current_item)

					if self.prop_info['use_prop_update']:
						bl_instance.on_prop_changed(self.bl_prop.name)

		# Reset Str Search
		elif value is Signal.ResetStrSearch:
			self.bl_prop_str_search.invalidate_nonpersist(bl_instance)

		# General __set__
		else:
			self.bl_prop.write(bl_instance, value)

			# Update Semantics
			if self.suppress_update.get(bl_instance.instance_id):
				self.suppress_update[bl_instance.instance_id] = False

			elif self.prop_info['use_prop_update']:
				bl_instance.on_prop_changed(self.bl_prop.name)

	####################
	# - Safe Callbacks
	####################
	def safe_str_cb(
		self, _self: bl_instance.BLInstance, context: bpy.types.Context, edit_text: str
	):
		"""Wrapper around `StringProperty.search` which keeps a non-persistent cache around search results.

		Reset by setting the descriptor to `Signal.ResetStrSearch`.
		"""
		cached_items = self.bl_prop_str_search.read_nonpersist(_self)
		if cached_items is not Signal.CacheNotReady:
			if cached_items is Signal.CacheEmpty:
				computed_items = self.str_cb(_self, context, edit_text)
				self.bl_prop_str_search.write_nonpersist(_self, computed_items)
				return computed_items
			return cached_items
		return []

	def safe_enum_cb(
		self, _self: bl_instance.BLInstance, context: bpy.types.Context
	) -> list[ct.BLEnumElement]:
		"""Wrapper around `EnumProperty.items` callback, which **guarantees** that returned strings will not be GCed by keeping a persistent + non-persistent cache.

		When a persistent cache exists, then the non-persistent cache will be filled at-will, since this is always guaranteed possible.
		Otherwise, the persistent cache will only be regenerated when `Signal.ResetEnumItems` is run.
		The original callback won't ever run other than then.

		Until then, `DEFAULT_ENUM_ITEMS_MANY` or `DEFAULT_ENUM_ITEMS_SINGLE` will be used as defaults (guaranteed to not dereference so long as the module is loaded).
		"""
		# Compute Value (if available)
		cached_items = self.bl_prop_enum_items.read_nonpersist(_self)
		if cached_items is Signal.CacheNotReady or cached_items is Signal.CacheEmpty:
			if _self is not None:
				persisted_items = self.bl_prop_enum_items.read(_self)
				if not persisted_items:
					computed_items = self.enum_cb(_self, context)
					_items = computed_items
				else:
					_items = persisted_items
			else:
				computed_items = self.enum_cb(_self, context)
				_items = computed_items

			# Fallback for Empty Persisted Items
			## -> Use [('NONE', ...)]
			## -> This guarantees that the enum items always has >=len(1)
			items = _items if _items else default_enum_items(self.bl_prop.is_enum_many)

			# Write Items -> Non-Persistent Cache
			self.bl_prop_enum_items.write_nonpersist(_self, items)
			return items
		return cached_items
