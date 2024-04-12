"""Implements various key caches on instances of Blender objects, especially nodes and sockets."""

import functools
import inspect
import typing as typ

import bpy
import msgspec
import sympy as sp
import sympy.physics.units as spu

from ...utils import extra_sympy_units as spux
from ...utils import logger
from . import contracts as ct
from . import managed_objs, sockets

log = logger.get(__name__)

InstanceID: typ.TypeAlias = str  ## Stringified UUID4


class BLInstance(typ.Protocol):
	"""An instance of a blender object, ex. nodes/sockets.

	Attributes:
		instance_id: Stringified UUID4 that uniquely identifies an instance, among all active instances on all active classes.
	"""

	instance_id: InstanceID


EncodableValue: typ.TypeAlias = typ.Any  ## msgspec-compatible
PropGetMethod: typ.TypeAlias = typ.Callable[[BLInstance], EncodableValue]
PropSetMethod: typ.TypeAlias = typ.Callable[[BLInstance, EncodableValue], None]

####################
# - (De)Serialization
####################
EncodedComplex: typ.TypeAlias = tuple[float, float] | list[float, float]
EncodedSympy: typ.TypeAlias = str
EncodedManagedObj: typ.TypeAlias = tuple[str, str] | list[str, str]
EncodedPydanticModel: typ.TypeAlias = tuple[str, str] | list[str, str]


def _enc_hook(obj: typ.Any) -> EncodableValue:
	"""Translates types not natively supported by `msgspec`, to an encodable form supported by `msgspec`.

	Parameters:
		obj: The object of arbitrary type to transform into an encodable value.

	Returns:
		A value encodable by `msgspec`.

	Raises:
		NotImplementedError: When the type transformation hasn't been implemented.
	"""
	if isinstance(obj, complex):
		return (obj.real, obj.imag)
	if isinstance(obj, sp.Basic | sp.MatrixBase | sp.Expr | spu.Quantity):
		return sp.srepr(obj)
	if isinstance(obj, managed_objs.ManagedObj):
		return (obj.name, obj.__class__.__name__)
	if isinstance(obj, ct.schemas.SocketDef):
		return (obj.model_dump(), obj.__class__.__name__)

	msg = f'Can\'t encode "{obj}" of type {type(obj)}'
	raise NotImplementedError(msg)


def _dec_hook(_type: type, obj: EncodableValue) -> typ.Any:
	"""Translates the `msgspec`-encoded form of an object back to its true form.

	Parameters:
		_type: The type to transform the `msgspec`-encoded object back into.
		obj: The encoded object of to transform back into an encodable value.

	Returns:
		A value encodable by `msgspec`.

	Raises:
		NotImplementedError: When the type transformation hasn't been implemented.
	"""
	if _type is complex and isinstance(obj, EncodedComplex):
		return complex(obj[0], obj[1])
	if (
		_type is sp.Basic
		and isinstance(obj, EncodedSympy)
		or _type is sp.Expr
		and isinstance(obj, EncodedSympy)
		or _type is sp.MatrixBase
		and isinstance(obj, EncodedSympy)
		or _type is spu.Quantity
		and isinstance(obj, EncodedSympy)
	):
		return sp.sympify(obj).subs(spux.ALL_UNIT_SYMBOLS)
	if (
		_type is managed_objs.ManagedBLMesh
		and isinstance(obj, EncodedManagedObj)
		or _type is managed_objs.ManagedBLImage
		and isinstance(obj, EncodedManagedObj)
		or _type is managed_objs.ManagedBLModifier
		and isinstance(obj, EncodedManagedObj)
	):
		return {
			'ManagedBLMesh': managed_objs.ManagedBLMesh,
			'ManagedBLImage': managed_objs.ManagedBLImage,
			'ManagedBLModifier': managed_objs.ManagedBLModifier,
		}[obj[1]](obj[0])
	if _type is ct.schemas.SocketDef:
		return getattr(sockets, obj[1])(**obj[0])

	msg = f'Can\'t decode "{obj}" to type {type(obj)}'
	raise NotImplementedError(msg)


ENCODER = msgspec.json.Encoder(enc_hook=_enc_hook, order='deterministic')

_DECODERS: dict[type, msgspec.json.Decoder] = {
	complex: msgspec.json.Decoder(type=complex, dec_hook=_dec_hook),
	sp.Basic: msgspec.json.Decoder(type=sp.Basic, dec_hook=_dec_hook),
	sp.Expr: msgspec.json.Decoder(type=sp.Expr, dec_hook=_dec_hook),
	sp.MatrixBase: msgspec.json.Decoder(type=sp.MatrixBase, dec_hook=_dec_hook),
	spu.Quantity: msgspec.json.Decoder(type=spu.Quantity, dec_hook=_dec_hook),
	managed_objs.ManagedBLMesh: msgspec.json.Decoder(
		type=managed_objs.ManagedBLMesh,
		dec_hook=_dec_hook,
	),
	managed_objs.ManagedBLImage: msgspec.json.Decoder(
		type=managed_objs.ManagedBLImage,
		dec_hook=_dec_hook,
	),
	managed_objs.ManagedBLModifier: msgspec.json.Decoder(
		type=managed_objs.ManagedBLModifier,
		dec_hook=_dec_hook,
	),
	# managed_objs.ManagedObj: msgspec.json.Decoder(
	# type=managed_objs.ManagedObj, dec_hook=_dec_hook
	# ),  ## Doesn't work b/c unions are not explicit
	ct.schemas.SocketDef: msgspec.json.Decoder(
		type=ct.schemas.SocketDef,
		dec_hook=_dec_hook,
	),
}
_DECODER_FALLBACK: msgspec.json.Decoder = msgspec.json.Decoder(dec_hook=_dec_hook)


@functools.cache
def DECODER(_type: type) -> msgspec.json.Decoder:  # noqa: N802
	"""Retrieve a suitable `msgspec.json.Decoder` by-type.

	Parameters:
		_type: The type to retrieve a decoder for.

	Returns:
		A suitable decoder.
	"""
	if (decoder := _DECODERS.get(_type)) is not None:
		return decoder

	return _DECODER_FALLBACK


def decode_any(_type: type, obj: str) -> typ.Any:
	naive_decode = DECODER(_type).decode(obj)
	if _type == dict[str, ct.schemas.SocketDef]:
		return {
			socket_name: getattr(sockets, socket_def_list[1])(**socket_def_list[0])
			for socket_name, socket_def_list in naive_decode.items()
		}

	log.critical(
		'Naive Decode of "%s" to "%s" (%s)', str(obj), str(naive_decode), str(_type)
	)
	return naive_decode


####################
# - Cache: Non-Persistent
####################
CACHE_NOPERSIST: dict[InstanceID, dict[typ.Any, typ.Any]] = {}


def invalidate_nonpersist_instance_id(instance_id: InstanceID) -> None:
	"""Invalidate any `instance_id` that might be utilizing cache space in `CACHE_NOPERSIST`.

	Note:
		This should be run by the `instance_id` owner in its `free()` method.

	Parameters:
		instance_id: The ID of the Blender object instance that's being freed.
	"""
	CACHE_NOPERSIST.pop(instance_id, None)


####################
# - Property Descriptor
####################
class CachedBLProperty:
	"""A descriptor that caches a computed attribute of a Blender node/socket/... instance (`bl_instance`), with optional cache persistence.

	Note:
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
			value = decode_any(self._type, encoded_value)
			cache_nopersist[self._bl_prop_name] = value
			return value

		# Compute Value
		## Fill Non-Persistent Cache
		## Fill Persistent Cache (maybe)
		value = self._getter_method(bl_instance)
		cache_nopersist[self._bl_prop_name] = value
		if self._persist:
			setattr(
				bl_instance, self._bl_prop_name, ENCODER.encode(value).decode('utf-8')
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

		Note:
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
			@bl_cache.cached(persist=True|False)
			def computed_prop(self) -> ...: return ...

		print(bl_instance.prop)  ## Computes first time
		print(bl_instance.prop)  ## Cached (maybe persistently in a property, maybe not)
	```

	When
	"""

	def decorator(getter_method: typ.Callable[[BLInstance], None]) -> type:
		return CachedBLProperty(getter_method=getter_method, persist=persist)

	return decorator


####################
# - Attribute Descriptor
####################
class BLField:
	"""A descriptor that allows persisting arbitrary types in Blender objects, with cached reads."""

	def __init__(self, default_value: typ.Any, triggers_prop_update: bool = True):
		"""Initializes and sets the attribute to a given default value.

		Parameters:
			default_value: The default value to use if the value is read before it's set.
			trigger_prop_update: Whether to run `bl_instance.sync_prop(attr_name)` whenever value is set.

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

		Note:
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
		encoded_default_value = ENCODER.encode(self._default_value).decode('utf-8')
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
			return decode_any(AttrType, getattr(_self, bl_attr_name))

		## Setter:
		## 1. Initialize bpy.props.StringProperty to Default (if undefined).
		## 3. Encode value (implicitly using the annotated type).
		## 2. Set bpy.props.StringProperty string.
		def setter(_self: BLInstance, value: AttrType) -> None:
			encoded_value = ENCODER.encode(value).decode('utf-8')
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
