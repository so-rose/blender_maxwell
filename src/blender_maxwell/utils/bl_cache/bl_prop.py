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

"""Defines `BLProp`, a high-level wrapper for interacting with Blender properties."""

import dataclasses
import functools
import typing as typ

from blender_maxwell.utils import bl_instance, logger

from . import managed_cache
from .bl_prop_type import BLPropInfo, BLPropType
from .signal import Signal

log = logger.get(__name__)


####################
# - Blender Property (Abstraction)
####################
@dataclasses.dataclass(kw_only=True, frozen=True)
class BLProp:
	"""A high-level wrapper encapsulating access to a Blender property.

	Attributes:
		name: The name of the Blender property, as one uses it.
		prop_info: Specifies the property's particular behavior, including subtype and UI.
		prop_type: The type to associate with the property.
			Especially relevant for structured deserialization.
		bl_prop_type: Identifier encapsulating which Blender property used for data storage, and how.
	"""

	name: str
	prop_info: BLPropInfo  ## TODO: Validate / Typing
	prop_type: type
	bl_prop_type: BLPropType

	####################
	# - Computed
	####################
	@functools.cached_property
	def bl_name(self):
		"""Deduces the actual attribute name at which the Blender property will be available."""
		return f'blfield__{self.name}'

	@functools.cached_property
	def enum_cache_key(self):
		"""Deduces an attribute name for use by the persistent cache component of `EnumProperty.items`.

		For dynamic enums, a persistent cache is not enough - a non-persistent cache must also be used to guarantee that returned strings will not dereference.
		**Letting dynamic enum strings dereference causes Blender to crash**.

		Use of a non-persistent cache alone introduces a massive startup burden, as _all_ of the potentially expensive `EnumProperty.items` methods must re-run.
		Should any depend on ex. internet connectivity, which is no longer available, elaborate failure modes may trigger.

		By using this key, we can persist `items` for re-caching on startup, to reap the benefits of both schemes and make dynamic `EnumProperty` usable in practice.
		"""
		return self.name + '__enum_cache'

	@functools.cached_property
	def str_cache_key(self):
		"""Deduce an internal name for string-search names distinct from the property name.

		Compared to dynamic enums, string-search names are very gentle.
		However, the mechanism is otherwise almost same, so similar logic makes a lot of sense.
		"""
		return self.name + '__str_cache'

	@functools.cached_property
	def display_name(self):
		"""Deduce a display name for the Blender property, assigned to the `name=` attribute."""
		return (
			'[JSON] ' if self.bl_prop_type == BLPropType.Serialized else ''
		) + f'BLField: {self.name}'

	@functools.cached_property
	def is_enum_many(self):
		return self.bl_prop_type in [BLPropType.SetEnum, BLPropType.SetDynEnum]

	####################
	# - Low-Level Methods
	####################
	def encode(self, value: typ.Any):
		"""Encode a value for compatibility with this Blender property, using the encapsulated types.

		A convenience method for `BLPropType.encode()`.
		"""
		return self.bl_prop_type.encode(value)

	@functools.cached_property
	def default_value(self) -> typ.Any:
		return self.prop_info.get('default')

	def decode(self, value: typ.Any):
		"""Encode a value for compatibility with this Blender property, using the encapsulated types.

		A convenience method for `BLPropType.decode()`.
		"""
		return self.bl_prop_type.decode(value, self.prop_type)

	####################
	# - Initialization
	####################
	def init_bl_type(
		self,
		bl_type: type[bl_instance.BLInstance],
		depends_on: frozenset[str] = frozenset(),
		enum_depends_on: frozenset[str] | None = None,
		strsearch_depends_on: frozenset[str] | None = None,
	) -> None:
		"""Declare the Blender property on a Blender class, ensuring that the property will be available to all `bl_instance.BLInstance` respecting instances of that class.

		- **Declare BLField**: Runs `bl_type.declare_blfield()` to ensure that `on_prop_changed` will invalidate the cache for this property.
		- **Set Property**: Runs `bl_type.set_prop()` to ensure that the Blender property will be available on instances of `bl_type`.

		Parameters:
			obj_type: The exact object type that will be stored in the Blender property.
				**Must** be chosen such that `BLPropType.from_type(obj_type) == self`.
		"""
		# Parse KWArgs for Blender Property
		kwargs_prop = self.bl_prop_type.parse_kwargs(
			self.prop_type,
			self.prop_info,
		)

		# Set Blender Property
		bl_type.declare_blfield(
			self.name,
			self.bl_name,
			use_dynamic_enum=self.prop_info.get('enum_dynamic', False),
			use_str_search=self.prop_info.get('str_search', False),
		)
		bl_type.set_prop(
			self.bl_name,
			self.bl_prop_type.bl_prop,
			# Property Options
			name=self.display_name,
			**kwargs_prop,
		)  ## TODO: Parse __doc__ for property descs

		for src_prop_name in depends_on:
			bl_type.declare_blfield_dep(src_prop_name, self.name)

		if self.prop_info.get('enum_dynamic', False) and enum_depends_on is not None:
			for src_prop_name in enum_depends_on:
				bl_type.declare_blfield_dep(
					src_prop_name, self.name, method='reset_enum'
				)

		if self.prop_info.get('str_search', False) and strsearch_depends_on is not None:
			for src_prop_name in strsearch_depends_on:
				bl_type.declare_blfield_dep(
					src_prop_name, self.name, method='reset_strsearch'
				)

	####################
	# - Instance Methods
	####################
	def read_nonpersist(self, bl_instance: bl_instance.BLInstance | None) -> typ.Any:
		"""Read the non-persistent cache value for this property.

		Returns:
			Generally, the cache value, with two exceptions.

			- `Signal.CacheNotReady`: When either `bl_instance` is None, or it doesn't yet have a unique `bl_instance.instance_id`.
				Indicates that the instance is not yet ready for use.
				For nodes, `init()` has not yet run.
				For sockets, `preinit()` has not yet run.

			- `Signal.CacheEmpty`: When the cache has no entry.
				A good idea might be to fill it immediately with `self.write_nonpersist(bl_instance)`.
		"""
		return managed_cache.read(
			bl_instance,
			self.bl_name,
			use_nonpersist=True,
			use_persist=False,
		)

	def read(self, bl_instance: bl_instance.BLInstance) -> typ.Any:
		"""Read the Blender property's particular value on the given `bl_instance`."""
		persisted_value = self.decode(
			managed_cache.read(
				bl_instance,
				self.bl_name,
				use_nonpersist=False,
				use_persist=True,
			)
		)
		if persisted_value is not Signal.CacheEmpty:
			return persisted_value

		msg = f"{self.name}: Can't read BLProp from instance {bl_instance}"
		raise ValueError(msg)

	def write(self, bl_instance: bl_instance.BLInstance, value: typ.Any) -> None:
		managed_cache.write(
			bl_instance,
			self.bl_name,
			self.encode(value),
			use_nonpersist=False,
			use_persist=True,
		)
		self.write_nonpersist(bl_instance, value)

	def write_nonpersist(
		self, bl_instance: bl_instance.BLInstance, value: typ.Any
	) -> None:
		managed_cache.write(
			bl_instance,
			self.bl_name,
			value,
			use_nonpersist=True,
			use_persist=False,
		)

	def invalidate_nonpersist(self, bl_instance: bl_instance.BLInstance | None) -> None:
		managed_cache.invalidate_nonpersist(
			bl_instance,
			self.bl_name,
		)
