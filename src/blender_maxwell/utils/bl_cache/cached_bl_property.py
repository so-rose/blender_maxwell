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

import contextlib
import inspect
import typing as typ

from blender_maxwell.utils import bl_instance, logger, serialize

from .bl_prop import BLProp
from .bl_prop_type import BLPropType
from .signal import Signal

log = logger.get(__name__)

####################
# - Types
####################
PropGetMethod: typ.TypeAlias = typ.Callable[
	[bl_instance.BLInstance], serialize.NaivelyEncodableType
]
PropSetMethod: typ.TypeAlias = typ.Callable[
	[bl_instance.BLInstance, serialize.NaivelyEncodableType], None
]


####################
# - CachedBLProperty
####################
class CachedBLProperty:
	"""A descriptor that caches a computed attribute of a Blender node/socket/... instance (`bl_instance`).

	Generally used via the associated decorator, `cached_bl_property`.

	Notes:
		It's like `@cached_property`, but on each Blender Instance ID.

	Attributes:
		getter_method: Method of `bl_instance` that computes the value.
		setter_method: Method of `bl_instance` that sets the value.
	"""

	def __init__(
		self,
		getter_method: PropGetMethod,
		persist: bool = False,
		depends_on: frozenset[str] = frozenset(),
	):
		"""Initialize the getter of the cached property.

		Parameters:
			getter_method: Method of `bl_instance` that computes the value.
		"""
		self.getter_method: PropGetMethod = getter_method
		self.setter_method: PropSetMethod | None = None

		self.persist: bool = persist
		self.depends_on: frozenset[str] = depends_on

		self.bl_prop: BLProp | None = None

		self.decode_type: type = inspect.signature(getter_method).return_annotation

		# Write Suppressing
		self.suppressed_update: dict[str, bool] = {}

		# Check Non-Empty Type Annotation
		## For now, just presume that all types can be encoded/decoded.
		if self.decode_type is inspect.Signature.empty:
			msg = f'A CachedBLProperty was instantiated, but its getter method "{self.getter_method}" has no return type annotation'
			raise TypeError(msg)

	def __set_name__(self, owner: type[bl_instance.BLInstance], name: str) -> None:
		"""Generates the property name from the name of the attribute that this descriptor is assigned to.

		Notes:
			- Run by Python when setting an instance of this class to an attribute.

		Parameters:
			owner: The class that contains an attribute assigned to an instance of this descriptor.
			name: The name of the attribute that an instance of descriptor was assigned to.
		"""
		self.bl_prop = BLProp(
			name=name,
			prop_info={'use_prop_update': True},
			prop_type=self.decode_type,
			bl_prop_type=BLPropType.Serialized,
		)
		self.bl_prop.init_bl_type(owner, depends_on=self.depends_on)

	def __get__(
		self,
		bl_instance: bl_instance.BLInstance | None,
		owner: type[bl_instance.BLInstance],
	) -> typ.Any:
		"""Retrieves the property from a cache, or computes it and fills the cache(s).

		Parameters:
			bl_instance: The Blender object this prop
		"""
		cached_value = self.bl_prop.read_nonpersist(bl_instance)
		if cached_value is Signal.CacheNotReady or cached_value is Signal.CacheEmpty:
			if bl_instance is not None:
				if self.persist:
					value = self.bl_prop.read(bl_instance)
				else:
					value = self.getter_method(bl_instance)

				self.bl_prop.write_nonpersist(bl_instance, value)
				return value
			return Signal.CacheNotReady
		return cached_value

	@contextlib.contextmanager
	def suppress_update(self, bl_instance: bl_instance.BLInstance) -> None:
		"""A context manager that suppresses all calls to `on_prop_changed()` for fields of the given `bl_instance` while active.

		Any change to a `BLProp` managed by this descriptor inevitably trips `bl_instance.on_bl_prop_changed()`.
		In response to these changes, `bl_instance.on_bl_prop_changed()` always signals the `Signal.InvalidateCache` via this descriptor.
		Unless something interferes, this results in a call to `bl_instance.on_prop_changed()`.

		Usually, this is great.
		But sometimes, like when ex. refreshing enum items, we **want** to be able to set the value of the `BLProp` **without** triggering that `bl_instance.on_prop_changed()`.
		By default, there is absolutely no way to accomplish this.

		That's where this context manager comes into play.
		While active, all calls to `bl_instance.on_prop_changed()` will be ignored for the given `bl_instance`, allowing us to freely set persistent properties without side effects.

		Examples:
			A simple illustrative example could look something like:

			```python
			with self.suppress_update(bl_instance):
				self.bl_prop.write(bl_instance, 'I won't trigger an update')

			self.bl_prop.write(bl_instance, 'I will trigger an update')
			```
		"""
		self.suppressed_update[bl_instance.instance_id] = True
		try:
			yield
		finally:
			self.suppressed_update[bl_instance.instance_id] = False
			## -> We could .pop(None).
			## -> But keeping a reused memory location around is GC friendly.

	def __set__(
		self, bl_instance: bl_instance.BLInstance | None, value: typ.Any
	) -> None:
		"""Runs the user-provided setter, after invalidating the caches.

		Notes:
			- This invalidates all caches without re-filling them.
			- The caches will be re-filled on the first `__get__` invocation, which may be slow due to having to run the getter method.

		Parameters:
			bl_instance: The Blender object this prop
		"""
		# Invalidate Cache
		## -> This empties the non-persistent cache.
		## -> If persist=True, this also writes the persistent cache (no update).
		## The 'on_prop_changed' method on the bl_instance might also be called.
		if value is Signal.InvalidateCache or value is Signal.InvalidateCacheNoUpdate:
			# Invalidate Partner Non-Persistent Caches
			## -> Only for the invalidation case do we also invalidate partners.
			if bl_instance is not None:
				# Fill Caches
				## -> persist=True: Fill Persist and Non-Persist Cache
				## -> persist=False: Fill Non-Persist Cache
				if not self.suppressed_update.get(bl_instance.instance_id, False):
					if self.persist:
						with self.suppress_update(bl_instance):
							self.bl_prop.write(
								bl_instance, self.getter_method(bl_instance)
							)

					else:
						self.bl_prop.write_nonpersist(
							bl_instance, self.getter_method(bl_instance)
						)

				# Trigger Update
				## -> Use InvalidateCacheNoUpdate to explicitly disable update.
				## -> If 'suppress_update' context manager is active, don't update.
				if value is Signal.InvalidateCache and not self.suppressed_update.get(
					bl_instance.instance_id, False
				):
					bl_instance.on_prop_changed(self.bl_prop.name)

		# Call Setter
		elif self.setter_method is not None:
			if bl_instance is not None:
				# Run Setter
				## -> The user-provided setter can set values as it sees fit.
				## -> The user-provided setter will not immediately trigger updates.
				with self.suppress_update(bl_instance):
					self.setter_method(bl_instance, value)

				# Fill Caches
				## -> persist=True: Fill Persist and Non-Persist Cache
				## -> persist=False: Fill Non-Persist Cache
				if self.persist:
					with self.suppress_update(bl_instance):
						self.bl_prop.write(bl_instance, self.getter_method(bl_instance))

				else:
					self.bl_prop.write_nonpersist(
						bl_instance, self.getter_method(bl_instance)
					)

				# Trigger Update
				## -> If 'suppress_update' context manager is active, don't update.
				if not self.suppressed_update.get(bl_instance.instance_id):
					bl_instance.on_prop_changed(self.bl_prop.name)

		else:
			msg = f'Tried to set "{value}" to "{self.prop_name}" on "{bl_instance.bl_label}", but a setter was not defined'
			raise NotImplementedError(msg)

	def setter(self, setter_method: PropSetMethod) -> typ.Self:
		"""Decorator to add a setter to the cached property.

		Returns:
			The same descriptor, so that use of the same method name for defining a setter won't change the semantics of the attribute.

		Examples:
			Without the decorator, it looks like this:
			```python
			class Test(bpy.types.Node):
				bl_label = 'Default'
				...
				def method(self) -> str: return self.bl_label
				attr = CachedBLProperty(getter_method=method)

				@attr.setter
				def attr(self, value: str) -> None:
					self.bl_label = 'Altered'
			```
		"""
		# Validate Setter Signature
		setter_sig = inspect.signature(setter_method)

		## Parameter Length
		if (sig_len := len(setter_sig.parameters)) != 2:  # noqa: PLR2004
			msg = f'Setter method for "{self.prop_name}" should have 2 parameters, not "{sig_len}"'
			raise TypeError(msg)

		## Parameter Value Type
		if (sig_ret_type := setter_sig.return_annotation) is not None:
			msg = f'Setter method for "{self.prop_name}" return value type "{sig_ret_type}", but it should be "None" (omitting an annotation does not imply "None")'
			raise TypeError(msg)

		self.setter_method = setter_method
		return self


####################
# - Decorator
####################
def cached_bl_property(
	persist: bool = False,
	depends_on: frozenset[str] = frozenset(),
):
	"""Decorator creating a descriptor that caches a computed attribute of a Blender node/socket.

	Many such `bl_instance`s rely on fast access to computed, cached properties, for example to ensure that `draw()` remains effectively non-blocking.

	Notes:
		- Unfortunately, `functools.cached_property` doesn't work.
		- Use `cached_attribute` if not using a node/socket.

	Examples:
	```python
		class CustomNode(bpy.types.Node):
			@bl_cache.cached()
			def computed_prop(self) -> ...: return ...

		print(bl_instance.prop)  ## Computes first time
		print(bl_instance.prop)  ## Cached (after restart, will recompute)
	```
	"""

	def decorator(getter_method: typ.Callable[[bl_instance.BLInstance], None]) -> type:
		return CachedBLProperty(
			getter_method=getter_method, persist=persist, depends_on=depends_on
		)

	return decorator
