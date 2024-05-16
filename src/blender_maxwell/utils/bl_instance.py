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

import typing as typ
import uuid
from types import MappingProxyType

import bpy

from blender_maxwell.utils import bl_cache, logger

InstanceID: typ.TypeAlias = str  ## Stringified UUID4

log = logger.get(__name__)


class BLInstance:
	"""An instance of a blender object, ex. nodes/sockets.

	Used as a common base of functionality for nodes/sockets, especially when it comes to the magic introduced by `bl_cache`.

	Notes:
		All the `@classmethod`s are designed to be invoked with `cls` as the subclass of `BLInstance`, not `BLInstance` itself.

		For practical reasons, introducing a metaclass here is not a good idea, and thus `abc.ABC` can't be used.
		To this end, only `self.on_prop_changed` needs a subclass implementation.
		It's a little sharp, but managable.

		Inheritance schemes like this are generally not enjoyable.
		However, the way Blender's node/socket classes are structured makes it the most practical way design for the functionality encapsulated here.

	Attributes:
		instance_id: Stringified UUID4 that uniquely identifies an instance, among all active instances on all active classes.
	"""

	####################
	# - Attributes
	####################
	instance_id: bpy.props.StringProperty(default='')

	blfields: typ.ClassVar[dict[str, str]] = MappingProxyType({})
	blfield_deps: typ.ClassVar[dict[str, list[str]]] = MappingProxyType({})

	blfields_dynamic_enum: typ.ClassVar[set[str]] = frozenset()
	blfield_dynamic_enum_deps: typ.ClassVar[dict[str, list[str]]] = MappingProxyType({})

	blfields_str_search: typ.ClassVar[set[str]] = frozenset()
	blfield_str_search_deps: typ.ClassVar[dict[str, list[str]]] = MappingProxyType({})

	####################
	# - Runtime Instance Management
	####################
	def reset_instance_id(self) -> None:
		"""Reset the Instance ID of a BLInstance.

		The Instance ID is used to index the instance-specific cache, since Blender doesn't always directly support keeping changing data on node/socket instances.

		Notes:
			Should be run whenever the instance is copied, so that the copy will index its own cache.

			The Instance ID is a `UUID4`, which is globally unique, negating the need for extraneous overlap-checks.
		"""
		self.instance_id = str(uuid.uuid4())

	@classmethod
	def assert_attrs_valid(cls, mandatory_props: set[str]) -> None:
		"""Asserts that all mandatory attributes are defined on the class.

		The list of mandatory objects is generally sourced from a global variable, `MANDATORY_PROPS`, which should be passed to this function while running `__init_subclass__`.

		Raises:
			ValueError: If a mandatory attribute defined in `base.MANDATORY_PROPS` is not defined on the class.
		"""
		for cls_attr in mandatory_props:
			if not hasattr(cls, cls_attr):
				msg = f'Node class {cls} does not define mandatory attribute "{cls_attr}".'
				raise ValueError(msg)

	####################
	# - Field Registration
	####################
	@classmethod
	def declare_blfield(
		cls,
		attr_name: str,
		bl_attr_name: str,
		use_dynamic_enum: bool = False,
		use_str_search: bool = False,
	) -> None:
		"""Declare the existance of a (cached) field and any properties affecting its invalidation.

		Primarily, the `attr_name -> bl_attr_name` map will be available via the `cls.blfields` dictionary.
		Thus, for use in UIs (where `bl_attr_name` must be used), one can use `cls.blfields[attr_name]`.

		Parameters:
			attr_name: The name of the attribute accessible via the instance.
			bl_attr_name: The name of the attribute containing the Blender property.
				This is used both as a persistant cache for `attr_name`, as well as (possibly) the data altered by the user from the UI.
			use_dynamic_enum: Will mark `attr_name` as a dynamic enum.
				Allows `self.regenerate_dynamic_field_persistance` to reset this property, whenever all dynamic `EnumProperty`s are reset at once.
			use_str_searc: The name of the attribute containing the Blender property.
				Allows `self.regenerate_dynamic_field_persistance` to reset this property, whenever all searched `StringProperty`s are reset at once.
		"""
		cls.blfields = cls.blfields | {attr_name: bl_attr_name}

		if use_dynamic_enum:
			cls.blfields_dynamic_enum = cls.blfields_dynamic_enum | {attr_name}

		if use_str_search:
			cls.blfields_str_search = cls.blfields_str_search | {attr_name}

	@classmethod
	def declare_blfield_dep(
		cls,
		src_prop_name: str,
		dst_prop_name: str,
		method: typ.Literal[
			'invalidate', 'reset_enum', 'reset_strsearch'
		] = 'invalidate',
	) -> None:
		"""Declare that `prop_name` relies on another property.

		This is critical for cached, computed properties that must invalidate their cache whenever any of the data they rely on changes.
		In practice, a chain of invalidation emerges naturally when this is put together, managed internally for performance.

		Notes:
			If the relevant `*_deps` dictionary is not defined on `cls`, we manually create it.
			This shadows the relevant `BLInstance` attribute, which is an immutable `MappingProxyType` on purpose, precisely to prevent the situation of altering data that shouldn't be common to all classes inheriting from `BLInstance`.

			Not clean, but it works.

		Parameters:
			dep_prop_name: The property that should, whenever changed, also invalidate the cache of `prop_name`.
			prop_name: The property that relies on another property.
		"""
		match method:
			case 'invalidate':
				if not cls.blfield_deps:
					cls.blfield_deps = {}
				deps = cls.blfield_deps
			case 'reset_enum':
				if not cls.blfield_dynamic_enum_deps:
					cls.blfield_dynamic_enum_deps = {}
				deps = cls.blfield_dynamic_enum_deps
			case 'reset_strsearch':
				if not cls.blfield_str_search_deps:
					cls.blfield_str_search_deps = {}
				deps = cls.blfield_str_search_deps

		if deps.get(src_prop_name) is None:
			deps[src_prop_name] = []

		deps[src_prop_name].append(dst_prop_name)

	@classmethod
	def set_prop(
		cls,
		bl_prop_name: str,
		prop: bpy.types.Property,
		**kwargs,
	) -> None:
		"""Adds a Blender property via `__annotations__`, so that it will be initialized on all subclasses.

		**All Blender properties trigger an update method** when updated from the UI, in order to invalidate the non-persistent cache of the associated `BLField`.
		Specifically, this behavior happens in `on_bl_prop_changed()`.

		However, whether anything else happens after that invalidation is entirely up to the particular `BLField`.
		Thus, `BLField` is put in charge of how/when updates occur.

		Notes:
			In general, Blender properties can't be set on classes directly.
			They must be added as type annotations, which Blender will read and understand.

			This is essentially a convenience method to encapsulate this unexpected behavior, as well as constrain the behavior of the `update` method somewhat.

		Parameters:
			bl_prop_name: The name of the property to set, as accessible from Blender.
				Generally, from code, the user would access the wrapping `BLField` instead of directly accessing the `bl_prop_name` attribute.
			prop: The `bpy.types.Property` to instantiate and attach..
			kwargs: Constructor arguments to pass to the Blender property.
				There are many mostly-documented nuances with these.
				The methods of `bl_cache.BLPropType` are designed to provide more strict, helpful abstractions for practical use.
		"""
		cls.__annotations__[bl_prop_name] = prop(
			update=lambda self, context: self.on_bl_prop_changed(bl_prop_name, context),
			**kwargs,
		)

	####################
	# - Runtime Field Management
	####################
	def regenerate_dynamic_field_persistance(self):
		"""Regenerate the persisted data of all dynamic enums and str search BLFields.

		In practice, this sets special "signal" values:
		- **Dynamic Enums**: The signal value `bl_cache.Signal.ResetEnumItems` will be set, causing `BLField.__set__` to regenerate the enum items using the user-provided callback.
		- **Searched Strings**: The signal value `bl_cache.Signal.ResetStrSearch` will be set, causing `BLField.__set__` to regenerate the available search strings using the user-provided callback.
		"""
		# Generate Enum Items
		## -> This guarantees that the items are persisted from the start.
		for dyn_enum_prop_name in self.blfields_dynamic_enum:
			setattr(self, dyn_enum_prop_name, bl_cache.Signal.ResetEnumItems)

		# Generate Str Search Items
		## -> Match dynamic enum semantics
		for str_search_prop_name in self.blfields_str_search:
			setattr(self, str_search_prop_name, bl_cache.Signal.ResetStrSearch)

	def on_bl_prop_changed(self, bl_prop_name: str, _: bpy.types.Context) -> None:
		"""Called when a property has been updated via the Blender UI.

		The only effect is to invalidate the non-persistent cache of the associated BLField.
		The BLField then decides whether to take any other action, ex. calling `self.on_prop_changed()`.
		"""
		## TODO: What about non-Blender set properties?

		# Strip the Internal Prefix
		## -> TODO: This is a bit of a hack. Use a contracts constant.
		prop_name = bl_prop_name.removeprefix('blfield__')
		# log.debug(
		# 'Callback on Property %s (stripped: %s)',
		# bl_prop_name,
		# prop_name,
		# )
		# log.debug(
		# 'Dependencies (PROP: %s) (ENUM: %s) (SEAR: %s)',
		# self.blfield_deps,
		# self.blfield_dynamic_enum_deps,
		# self.blfield_str_search_deps,
		# )

		# Invalidate Property Cache
		## -> Only the non-persistent cache is regenerated.
		## -> The BLField decides whether to trigger `on_prop_changed`.
		if prop_name in self.blfields:
			# RULE: =1 DataChanged per Dependency Chain
			## -> We MUST invalidate the cache, but might not want to update.
			## -> Update should only be triggered when ==0 dependents.
			setattr(self, prop_name, bl_cache.Signal.InvalidateCacheNoUpdate)

			# Invalidate Dependent Properties (incl. DynEnums and StrSearch)
			## -> NOTE: Dependent props may also trigger `on_prop_changed`.
			## -> Meaning, don't use extraneous dependencies (as usual).
			for deps, invalidate_signal in zip(
				[
					self.blfield_deps,
					self.blfield_dynamic_enum_deps,
					self.blfield_str_search_deps,
				],
				[
					bl_cache.Signal.InvalidateCache,
					bl_cache.Signal.ResetEnumItems,
					bl_cache.Signal.ResetStrSearch,
				],
				strict=True,
			):
				if prop_name in deps:
					for dst_prop_name in deps[prop_name]:
						# log.debug(
						# 'Property %s is invalidating %s',
						# prop_name,
						# dst_prop_name,
						# )
						setattr(
							self,
							dst_prop_name,
							invalidate_signal,
						)

			# Do Update AFTER Dependencies
			## -> Yes, update will run once per dependency.
			## -> Don't abuse dependencies :)
			## -> If no-update is important, use_prop_update is still respected.
			setattr(self, prop_name, bl_cache.Signal.DoUpdate)

	def on_prop_changed(self, prop_name: str) -> None:
		"""Triggers changes/an event chain based on a changed property.

		In general, the `BLField` descriptor associated with `prop_name` decides whether this method should be called whenever `__set__` is used.
		An indirect consequence of this is that `self.on_bl_prop_changed`, which is _always_ triggered, may only _sometimes_ result in `on_prop_changed` being called, at the discretion of the relevant `BLField`.

		Notes:
			**Must** be overridden on all `BLInstance` subclasses.
		"""
		raise NotImplementedError