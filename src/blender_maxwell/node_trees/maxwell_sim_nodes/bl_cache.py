"""Implements various key caches on instances of Blender objects, especially nodes and sockets."""

import functools
import inspect
import typing as typ

import bpy

from ...utils import logger, serialize

log = logger.get(__name__)

InstanceID: typ.TypeAlias = str  ## Stringified UUID4


class BLInstance(typ.Protocol):
	"""An instance of a blender object, ex. nodes/sockets.

	Attributes:
		instance_id: Stringified UUID4 that uniquely identifies an instance, among all active instances on all active classes.
	"""

	instance_id: InstanceID

	@classmethod
	def set_prop(
		cls,
		prop_name: str,
		prop: bpy.types.Property,
		no_update: bool = False,
		update_with_name: str | None = None,
		**kwargs,
	) -> None: ...


PropGetMethod: typ.TypeAlias = typ.Callable[[BLInstance], serialize.EncodableValue]
PropSetMethod: typ.TypeAlias = typ.Callable[
	[BLInstance, serialize.EncodableValue], None
]


####################
# - Cache: Non-Persistent
####################
CACHE_NOPERSIST: dict[InstanceID, dict[typ.Any, typ.Any]] = {}


def invalidate_nonpersist_instance_id(instance_id: InstanceID) -> None:
	"""Invalidate any `instance_id` that might be utilizing cache space in `CACHE_NOPERSIST`.

	Notes:
		This should be run by the `instance_id` owner in its `free()` method.

	Parameters:
		instance_id: The ID of the Blender object instance that's being freed.
	"""
	CACHE_NOPERSIST.pop(instance_id, None)


####################
# - Property Descriptor
####################
class KeyedCache:
	def __init__(
		self,
		func: typ.Callable,
		exclude: set[str],
		encode: set[str],
	):
		# Function Information
		self.func: typ.Callable = func
		self.func_sig: inspect.Signature = inspect.signature(self.func)

		# Arg -> Key Information
		self.exclude: set[str] = exclude
		self.include: set[str] = set(self.func_sig.parameters.keys()) - exclude
		self.encode: set[str] = encode

		# Cache Information
		self.key_schema: tuple[str, ...] = tuple(
			[
				arg_name
				for arg_name in self.func_sig.parameters
				if arg_name not in exclude
			]
		)
		self.caches: dict[str | None, dict[tuple[typ.Any, ...], typ.Any]] = {}

	@property
	def is_method(self):
		return 'self' in self.exclude

	def cache(self, instance_id: str | None) -> dict[tuple[typ.Any, ...], typ.Any]:
		if self.caches.get(instance_id) is None:
			self.caches[instance_id] = {}

		return self.caches[instance_id]

	def _encode_key(self, arguments: dict[str, typ.Any]):
		## WARNING: Order of arguments matters. Arguments may contain 'exclude'd elements.
		return tuple(
			[
				(
					arg_value
					if arg_name not in self.encode
					else serialize.encode(arg_value)
				)
				for arg_name, arg_value in arguments.items()
				if arg_name in self.include
			]
		)

	def __get__(
		self, bl_instance: BLInstance | None, owner: type[BLInstance]
	) -> typ.Callable:
		_func = functools.partial(self, bl_instance)
		_func.invalidate = functools.partial(
			self.__class__.invalidate, self, bl_instance
		)
		return _func

	def __call__(self, *args, **kwargs):
		# Test Argument Bindability to Decorated Function
		try:
			bound_args = self.func_sig.bind(*args, **kwargs)
		except TypeError as ex:
			msg = f'Can\'t bind arguments (args={args}, kwargs={kwargs}) to @keyed_cache-decorated function "{self.func.__name__}" (signature: {self.func_sig})"'
			raise ValueError(msg) from ex

		# Check that Parameters for Keying the Cache are Available
		bound_args.apply_defaults()
		all_arg_keys = set(bound_args.arguments.keys())
		if not self.include <= (all_arg_keys - self.exclude):
			msg = f'Arguments spanning the keyed cached ({self.include}) are not available in the non-excluded arguments passed to "{self.func.__name__}": {all_arg_keys - self.exclude}'
			raise ValueError(msg)

		# Create Keyed Cache Entry
		key = self._encode_key(bound_args.arguments)
		cache = self.cache(args[0].instance_id if self.is_method else None)
		if (value := cache.get(key)) is None:
			value = self.func(*args, **kwargs)
			cache[key] = value

		return value

	def invalidate(
		self, bl_instance: BLInstance | None, **arguments: dict[str, typ.Any]
	) -> dict[str, typ.Any]:
		# Determine Wildcard Arguments
		wildcard_arguments = {
			arg_name for arg_name, arg_value in arguments.items() if arg_value is ...
		}

		# Compute Keys to Invalidate
		arguments_hashable = {
			arg_name: serialize.encode(arg_value)
			if arg_name in self.encode and arg_name not in wildcard_arguments
			else arg_value
			for arg_name, arg_value in arguments.items()
		}
		cache = self.cache(bl_instance.instance_id if self.is_method else None)
		for key in list(cache.keys()):
			if all(
				arguments_hashable.get(arg_name) == arg_value
				for arg_name, arg_value in zip(self.key_schema, key, strict=True)
				if arg_name not in wildcard_arguments
			):
				cache.pop(key)


def keyed_cache(exclude: set[str], encode: set[str] = frozenset()) -> typ.Callable:
	def decorator(func: typ.Callable) -> typ.Callable:
		return KeyedCache(
			func,
			exclude=exclude,
			encode=encode,
		)

	return decorator


####################
# - Property Descriptor
####################
class CachedBLProperty:
	"""A descriptor that caches a computed attribute of a Blender node/socket/... instance (`bl_instance`), with optional cache persistence.

	Notes:
		**Accessing the internal `_*` attributes is likely an anti-pattern**.

		`CachedBLProperty` does not own the data; it only provides a convenient interface of running user-provided getter/setters.
		This also applies to the `bpy.types.Property` entry created by `CachedBLProperty`, which should not be accessed directly.

	Attributes:
		_getter_method: Method of `bl_instance` that computes the value.
		_setter_method: Method of `bl_instance` that sets the value.
		_persist: Whether to persist the value on a `bpy.types.Property` defined on `bl_instance`.
			The name of this `bpy.types.Property` will be `cache__<prop_name>`.
		_type: The type of the value, used by the persistent decoder.
	"""

	def __init__(self, getter_method: PropGetMethod, persist: bool):
		"""Initialize the getter (and persistance) of the cached property.

		Notes:
			- When `persist` is true, the return annotation of the getter mathod will be used to guide deserialization.

		Parameters:
			getter_method: Method of `bl_instance` that computes the value.
			persist: Whether to persist the value on a `bpy.types.Property` defined on `bl_instance`.
				The name of this `bpy.types.Property` will be `cache__<prop_name>`.
		"""
		self._getter_method: PropGetMethod = getter_method
		self._setter_method: PropSetMethod | None = None

		# Persistance
		self._persist: bool = persist
		self._type: type | None = (
			inspect.signature(getter_method).return_annotation if persist else None
		)

		# Check Non-Empty Type Annotation
		## For now, just presume that all types can be encoded/decoded.
		if self._type is not None and self._type is inspect.Signature.empty:
			msg = f'A CachedBLProperty was instantiated with "persist={persist}", but its getter method "{self._getter_method}" has no return type annotation'
			raise TypeError(msg)

	def __set_name__(self, owner: type[BLInstance], name: str) -> None:
		"""Generates the property name from the name of the attribute that this descriptor is assigned to.

		Notes:
			- Run by Python when setting an instance of this class to an attribute.

		Parameters:
			owner: The class that contains an attribute assigned to an instance of this descriptor.
			name: The name of the attribute that an instance of descriptor was assigned to.
		"""
		self.prop_name: str = name
		self._bl_prop_name: str = f'blcache__{name}'

		# Define Blender Property (w/Update Sync)
		owner.set_prop(
			self._bl_prop_name,
			bpy.props.StringProperty,
			name=f'DO NOT USE: Cache for {self.prop_name}',
			default='',
			no_update=True,
		)

	def __get__(
		self, bl_instance: BLInstance | None, owner: type[BLInstance]
	) -> typ.Any:
		"""Retrieves the property from a cache, or computes it and fills the cache(s).

		If `self._persist` is `True`, the persistent cache will be checked and filled after the non-persistent cache.

		Notes:
			- The persistent cache keeps the
			- The persistent cache is fast and has good compatibility (courtesy `msgspec` encoding), but isn't nearly as fast as

		Parameters:
			bl_instance: The Blender object this prop
		"""
		if bl_instance is None:
			return None

		# Create Non-Persistent Cache Entry
		## Prefer explicit cache management to 'defaultdict'
		if CACHE_NOPERSIST.get(bl_instance.instance_id) is None:
			CACHE_NOPERSIST[bl_instance.instance_id] = {}
		cache_nopersist = CACHE_NOPERSIST[bl_instance.instance_id]

		# Try Hit on Non-Persistent Cache
		if (value := cache_nopersist.get(self._bl_prop_name)) is not None:
			return value

		# Try Hit on Persistent Cache
		## Hit: Fill Non-Persistent Cache
		if (
			self._persist
			and (encoded_value := getattr(bl_instance, self._bl_prop_name)) != ''
		):
			value = serialize.decode(self._type, encoded_value)
			cache_nopersist[self._bl_prop_name] = value
			return value

		# Compute Value
		## Fill Non-Persistent Cache
		## Fill Persistent Cache (maybe)
		value = self._getter_method(bl_instance)
		cache_nopersist[self._bl_prop_name] = value
		if self._persist:
			setattr(
				bl_instance, self._bl_prop_name, serialize.encode(value).decode('utf-8')
			)
		return value

	def __set__(self, bl_instance: BLInstance, value: typ.Any) -> None:
		"""Runs the user-provided setter, after invalidating the caches.

		Notes:
			- This invalidates all caches without re-filling them.
			- The caches will be re-filled on the first `__get__` invocation, which may be slow due to having to run the getter method.

		Parameters:
			bl_instance: The Blender object this prop
		"""
		if self._setter_method is None:
			msg = f'Tried to set "{value}" to "{self.prop_name}" on "{bl_instance.bl_label}", but a setter was not defined'
			raise NotImplementedError(msg)

		# Invalidate Caches
		self._invalidate_cache(bl_instance)

		# Set the Value
		self._setter_method(bl_instance, value)

	def setter(self, setter_method: PropSetMethod) -> typ.Self:
		"""Decorator to add a setter to the cached property.

		Returns:
			The same descriptor, so that use of the same method name for defining a setter won't change the semantics of the attribute.

		Examples:
			Without the decor
			```python
			class Test(bpy.types.Node):
				bl_label = 'Default'
				...
				def method(self) -> str: return self.bl_label
				attr = CachedBLProperty(getter_method=method, persist=False)

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

		self._setter_method = setter_method
		return self

	def _invalidate_cache(self, bl_instance: BLInstance) -> None:
		"""Invalidates all caches that might be storing the computed property value.

		This is invoked by `__set__`.

		Notes:
			Will not delete the `bpy.props.StringProperty`; instead, it will be set to ''.

		Parameters:
			bl_instance: The instance of the Blender object that contains this property.

		Examples:
			It is discouraged to run this directly, as any use-pattern that requires manually invalidating a property cache is **likely an anti-pattern**.

			With that disclaimer, manual invocation looks like this:
			```python
			bl_instance.attr._invalidate_cache()
			```
		"""
		# Invalidate Non-Persistent Cache
		if CACHE_NOPERSIST.get(bl_instance.instance_id) is not None:
			CACHE_NOPERSIST[bl_instance.instance_id].pop(self._bl_prop_name, None)

		# Invalidate Persistent Cache
		if self._persist and getattr(bl_instance, self._bl_prop_name) != '':
			setattr(bl_instance, self._bl_prop_name, '')


## TODO: How do we invalidate the data that the computed cached property depends on?
####################
# - Property Decorators
####################
def cached_bl_property(persist: bool = ...):
	"""Decorator creating a descriptor that caches a computed attribute of a Blender node/socket.

	Many such `bl_instance`s rely on fast access to computed, cached properties, for example to ensure that `draw()` remains effectively non-blocking.
	It is also sometimes desired that this cache persist on `bl_instance`, ex. in the case of loose sockets or cached web data.

	Notes:
		- Unfortunately, `functools.cached_property` doesn't work, and can't handle persistance.
		- Use `cached_attribute` instead if merely persisting the value is desired.

	Parameters:
		persist: Whether or not to persist the cache value in the Blender object.
			This should be used when the **source(s) of the computed value also persists with the Blender object**.
			For example, this is especially helpful when caching information for use in `draw()` methods, so that reloading the file won't alter the cache.

	Examples:
	```python
		class CustomNode(bpy.types.Node):
			@bl_cache.cached(persist=True)
			def computed_prop(self) -> ...: return ...

		print(bl_instance.prop)  ## Computes first time
		print(bl_instance.prop)  ## Cached (after restart, will read from persistent cache)
	```
	"""

	def decorator(getter_method: typ.Callable[[BLInstance], None]) -> type:
		return CachedBLProperty(getter_method=getter_method, persist=persist)

	return decorator


####################
# - Attribute Descriptor
####################
class BLField:
	"""A descriptor that allows persisting arbitrary types in Blender objects, with cached reads."""

	def __init__(
		self, default_value: typ.Any, triggers_prop_update: bool = True
	) -> typ.Self:
		"""Initializes and sets the attribute to a given default value.

		Parameters:
			default_value: The default value to use if the value is read before it's set.
			triggers_prop_update: Whether to run `bl_instance.sync_prop(attr_name)` whenever value is set.

		"""
		log.debug(
			'Initializing BLField (default_value=%s, triggers_prop_update=%s)',
			str(default_value),
			str(triggers_prop_update),
		)
		self._default_value: typ.Any = default_value
		self._triggers_prop_update: bool = triggers_prop_update

	def __set_name__(self, owner: type[BLInstance], name: str) -> None:
		"""Sets up getters/setters for attribute access, and sets up a `CachedBLProperty` to internally utilize them.

		Our getter/setter essentially reads/writes to a `bpy.props.StringProperty`, with

		and use them as user-provided getter/setter to internally define a normal non-persistent `CachedBLProperty`.
		As a result, we can reuse almost all of the logic in `CachedBLProperty`

		Notes:
			Run by Python when setting an instance of this class to an attribute.

		Parameters:
			owner: The class that contains an attribute assigned to an instance of this descriptor.
			name: The name of the attribute that an instance of descriptor was assigned to.
		"""
		# Compute Name and Type of Property
		## Also compute the internal
		attr_name = name
		bl_attr_name = f'blattr__{name}'
		if (AttrType := inspect.get_annotations(owner).get(name)) is None:  # noqa: N806
			msg = f'BLField "{self.prop_name}" must define a type annotation, but doesn\'t.'
			raise TypeError(msg)

		# Define Blender Property (w/Update Sync)
		encoded_default_value = serialize.encode(self._default_value).decode('utf-8')
		log.debug(
			'%s set to StringProperty w/default "%s" and no_update="%s"',
			bl_attr_name,
			encoded_default_value,
			str(not self._triggers_prop_update),
		)
		owner.set_prop(
			bl_attr_name,
			bpy.props.StringProperty,
			name=f'Encoded Attribute for {attr_name}',
			default=encoded_default_value,
			no_update=not self._triggers_prop_update,
			update_with_name=attr_name,
		)

		## Getter:
		## 1. Initialize bpy.props.StringProperty to Default (if undefined).
		## 2. Retrieve bpy.props.StringProperty string.
		## 3. Decode using annotated type.
		def getter(_self: BLInstance) -> AttrType:
			return serialize.decode(AttrType, getattr(_self, bl_attr_name))

		## Setter:
		## 1. Initialize bpy.props.StringProperty to Default (if undefined).
		## 3. Encode value (implicitly using the annotated type).
		## 2. Set bpy.props.StringProperty string.
		def setter(_self: BLInstance, value: AttrType) -> None:
			encoded_value = serialize.encode(value).decode('utf-8')
			log.debug(
				'Writing BLField attr "%s" w/encoded value: %s',
				bl_attr_name,
				encoded_value,
			)
			setattr(_self, bl_attr_name, encoded_value)

		# Initialize CachedBLProperty w/Getter and Setter
		## This is the usual descriptor assignment procedure.
		self._cached_bl_property = CachedBLProperty(getter_method=getter, persist=False)
		self._cached_bl_property.__set_name__(owner, name)
		self._cached_bl_property.setter(setter)

	def __get__(
		self, bl_instance: BLInstance | None, owner: type[BLInstance]
	) -> typ.Any:
		return self._cached_bl_property.__get__(bl_instance, owner)

	def __set__(self, bl_instance: BLInstance, value: typ.Any) -> None:
		self._cached_bl_property.__set__(bl_instance, value)
