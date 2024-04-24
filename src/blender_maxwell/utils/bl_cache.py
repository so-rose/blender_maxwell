"""Implements various key caches on instances of Blender objects, especially nodes and sockets."""

## TODO: Note that persist=True on cached_bl_property may cause a draw method to try and write to a Blender property, which Blender disallows.

import enum
import functools
import inspect
import typing as typ
import uuid
from pathlib import Path

import bpy

from blender_maxwell import contracts as ct
from blender_maxwell.utils import logger, serialize

log = logger.get(__name__)

InstanceID: typ.TypeAlias = str  ## Stringified UUID4


class Signal(enum.StrEnum):
	"""A value used to signal the descriptor via its `__set__`.

	Such a signal **must** be entirely unique: Even a well-thought-out string could conceivably produce a very nasty bug, where instead of setting a descriptor-managed attribute, the user would inadvertently signal the descriptor.

	To make it effectively impossible to confuse any other object whatsoever with a signal, the enum values are set to per-session `uuid.uuid4()`.

	Notes:
		**Do not** use this enum for anything other than directly signalling a `bl_cache` descriptor via its setter.

		**Do not** store this enum `Signal` in a variable or method binding that survives longer than the session.

		**Do not** persist this enum; the values will change whenever `bl_cache` is (re)loaded.
	"""

	InvalidateCache: str = str(uuid.uuid4())
	ResetEnumItems: str = str(uuid.uuid4())
	ResetStrSearch: str = str(uuid.uuid4())


class BLInstance(typ.Protocol):
	"""An instance of a blender object, ex. nodes/sockets.

	Attributes:
		instance_id: Stringified UUID4 that uniquely identifies an instance, among all active instances on all active classes.
	"""

	instance_id: InstanceID

	def reset_instance_id(self) -> None: ...

	@classmethod
	def declare_blfield(
		cls, attr_name: str, bl_attr_name: str, prop_ui: bool = False
	) -> None: ...

	@classmethod
	def set_prop(
		cls,
		prop_name: str,
		prop: bpy.types.Property,
		no_update: bool = False,
		update_with_name: str | None = None,
		**kwargs,
	) -> None: ...


class BLEnumStrEnum(typ.Protocol):
	@staticmethod
	def to_name(value: typ.Self) -> str: ...

	@staticmethod
	def to_icon(value: typ.Self) -> ct.BLIcon: ...


StringPropSubType: typ.TypeAlias = typ.Literal[
	'FILE_PATH', 'DIR_PATH', 'FILE_NAME', 'BYTE_STRING', 'PASSWORD', 'NONE'
]

StrMethod: typ.TypeAlias = typ.Callable[
	[BLInstance, bpy.types.Context, str], list[tuple[str, str]]
]
EnumMethod: typ.TypeAlias = typ.Callable[
	[BLInstance, bpy.types.Context], list[ct.BLEnumElement]
]

PropGetMethod: typ.TypeAlias = typ.Callable[
	[BLInstance], serialize.NaivelyEncodableType
]
PropSetMethod: typ.TypeAlias = typ.Callable[
	[BLInstance, serialize.NaivelyEncodableType], None
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
			- The non-persistent cache keeps the object in memory.
			- The persistent cache serializes the object and stores it as a string on the BLInstance. This is often fast enough, and has decent compatibility (courtesy `msgspec`), it isn't nearly as fast as the non-persistent cache, and there are gotchas.

		Parameters:
			bl_instance: The Blender object this prop
		"""
		if bl_instance is None:
			return None
		if not bl_instance.instance_id:
			log.debug(
				"Can't Get CachedBLProperty: Instance ID not (yet) defined on BLInstance %s",
				str(bl_instance),
			)
			return

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

	def __set__(self, bl_instance: BLInstance | None, value: typ.Any) -> None:
		"""Runs the user-provided setter, after invalidating the caches.

		Notes:
			- This invalidates all caches without re-filling them.
			- The caches will be re-filled on the first `__get__` invocation, which may be slow due to having to run the getter method.

		Parameters:
			bl_instance: The Blender object this prop
		"""
		if bl_instance is None:
			return
		if not bl_instance.instance_id:
			log.debug(
				"Can't Set CachedBLProperty: Instance ID not (yet) defined on BLInstance %s",
				str(bl_instance),
			)
			return

		if value == Signal.InvalidateCache:
			self._invalidate_cache(bl_instance)
			return

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
		"""
		# Invalidate Non-Persistent Cache
		if CACHE_NOPERSIST.get(bl_instance.instance_id) is not None:
			CACHE_NOPERSIST[bl_instance.instance_id].pop(self._bl_prop_name, None)

		# Invalidate Persistent Cache
		if self._persist and getattr(bl_instance, self._bl_prop_name) != '':
			setattr(bl_instance, self._bl_prop_name, '')


####################
# - Property Decorators
####################
def cached_bl_property(persist: bool = False):
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
		self,
		default_value: typ.Any = None,
		use_prop_update: bool = True,
		## Static
		prop_ui: bool = False,
		prop_flags: set[ct.BLPropFlag] | None = None,
		abs_min: int | float | None = None,
		abs_max: int | float | None = None,
		soft_min: int | float | None = None,
		soft_max: int | float | None = None,
		float_step: int | None = None,
		float_prec: int | None = None,
		str_secret: bool | None = None,
		path_type: typ.Literal['dir', 'file'] | None = None,
		## Static / Dynamic
		enum_many: bool | None = None,
		## Dynamic
		str_cb: StrMethod | None = None,
		enum_cb: EnumMethod | None = None,
	) -> typ.Self:
		"""Initializes and sets the attribute to a given default value.

		The attribute **must** declare a type annotation, and it **must** match the type of `default_value`.

		Parameters:
			default_value: The default value to use if the value is read before it's set.
			use_prop_update: Configures the BLField to run `bl_instance.on_prop_changed(attr_name)` whenever value is set.
				This is done by setting the `update` method.
			enum_cb: Method used to generate new enum elements whenever `Signal.ResetEnum` is presented.

		"""
		log.debug(
			'Initializing BLField (default_value=%s, use_prop_update=%s)',
			str(default_value),
			str(use_prop_update),
		)
		self._default_value: typ.Any = default_value
		self._use_prop_update: bool = use_prop_update

		## Static
		self._prop_ui = prop_ui
		self._prop_flags = prop_flags
		self._min = abs_min
		self._max = abs_max
		self._soft_min = soft_min
		self._soft_max = soft_max
		self._float_step = float_step
		self._float_prec = float_prec
		self._str_secret = str_secret
		self._path_type = path_type

		## Static / Dynamic
		self._enum_many = enum_many

		## Dynamic
		self._set_ser_default = False
		self._str_cb = str_cb
		self._enum_cb = enum_cb

		## HUGE TODO: Persist these
		self._str_cb_cache = {}
		self._enum_cb_cache = {}

	####################
	# - Safe Callbacks
	####################
	def _safe_str_cb(
		self, _self: BLInstance, context: bpy.types.Context, edit_text: str
	):
		"""Wrapper around StringProperty.search which **guarantees** that returned strings will not be garbage collected.

		Regenerate by passing `Signal.ResetStrSearch`.
		"""
		if self._str_cb_cache.get(_self.instance_id) is None:
			self._str_cb_cache[_self.instance_id] = self._str_cb(
				_self, context, edit_text
			)

		return self._str_cb_cache[_self.instance_id]

	def _safe_enum_cb(self, _self: BLInstance, context: bpy.types.Context):
		"""Wrapper around EnumProperty.items callback, which **guarantees** that returned strings will not be garbage collected.

		The mechanism is simple: The user-generated callback is run once, then cached in the descriptor instance for subsequent use.
		This guarantees that the user won't crash Blender by returning dynamically generated strings in the user-provided callback.

		The cost, however, is that user-provided callback won't run eagerly anymore.
		Thus, whenever the user wants the items in the enum to update, they must manually set the descriptor attribute to the value `Signal.ResetEnumItems`.
		"""
		if self._enum_cb_cache.get(_self.instance_id) is None:
			# Retrieve Dynamic Enum Items
			enum_items = self._enum_cb(_self, context)

			# Ensure len(enum_items) >= 1
			## There must always be one element to prevent invalid usage.
			if len(enum_items) == 0:
				self._enum_cb_cache[_self.instance_id] = [
					(
						'NONE',
						'None',
						'No items...',
						'',
						0 if not self._enum_many else 2**0,
					)
				]
			else:
				self._enum_cb_cache[_self.instance_id] = enum_items

		return self._enum_cb_cache[_self.instance_id]

	def __set_name__(self, owner: type[BLInstance], name: str) -> None:
		"""Sets up the descriptor on the class level, preparing it for per-instance use.

		- The type annotation of the attribute is noted, as it might later guide (de)serialization of the field.
		- An appropriate `bpy.props.Property` is chosen for the type annotaiton, with a default-case fallback of `bpy.props.StringProperty` containing serialized data.

		Our getter/setter essentially reads/writes to a `bpy.props.StringProperty`, with

		and use them as user-provided getter/setter to internally define a normal non-persistent `CachedBLProperty`.
		As a result, we can reuse almost all of the logic in `CachedBLProperty`

		Notes:
			Run by Python when setting an instance of this class to an attribute.

			For StringProperty subtypes, see: <https://blender.stackexchange.com/questions/104875/what-do-the-different-options-for-subtype-enumerator-in-stringproperty-do>

		Parameters:
			owner: The class that contains an attribute assigned to an instance of this descriptor.
			name: The name of the attribute that an instance of descriptor was assigned to.
		"""
		# Compute Name of Property
		## Internal name uses 'blfield__' to avoid unfortunate overlaps.
		attr_name = name
		bl_attr_name = f'blfield__{name}'

		owner.declare_blfield(attr_name, bl_attr_name, prop_ui=self._prop_ui)

		# Compute Type of Property
		## The type annotation of the BLField guides (de)serialization.
		if (AttrType := inspect.get_annotations(owner).get(name)) is None:
			msg = f'BLField "{self.prop_name}" must define a type annotation, but doesn\'t'
			raise TypeError(msg)

		# Define Blender Property (w/Update Sync)
		default_value = None
		no_default_value = False
		prop_is_serialized = False
		kwargs_prop = {}

		## Reusable Snippets
		def _add_min_max_kwargs():
			kwargs_prop |= {'min': self._abs_min} if self._abs_min is not None else {}
			kwargs_prop |= {'max': self._abs_max} if self._abs_max is not None else {}
			kwargs_prop |= (
				{'soft_min': self._soft_min} if self._soft_min is not None else {}
			)
			kwargs_prop |= (
				{'soft_max': self._soft_max} if self._soft_max is not None else {}
			)

		def _add_float_kwargs():
			kwargs_prop |= (
				{'step': self._float_step} if self._float_step is not None else {}
			)
			kwargs_prop |= (
				{'precision': self._float_prec} if self._float_prec is not None else {}
			)

		## Property Flags
		kwargs_prop |= {
			'options': self._prop_flags if self._prop_flags is not None else set()
		}

		## Scalar Bool
		if AttrType is bool:
			default_value = self._default_value
			BLProp = bpy.props.BoolProperty

		## Scalar Int
		elif AttrType is int:
			default_value = self._default_value
			BLProp = bpy.props.IntProperty
			_add_min_max_kwargs()

		## Scalar Float
		elif AttrType is float:
			default_value = self._default_value
			BLProp = bpy.props.FloatProperty
			_add_min_max_kwargs()
			_add_float_kwargs()

		## Vector Bool
		elif typ.get_origin(AttrType) is tuple and all(
			T is bool for T in typ.get_args(AttrType)
		):
			default_value = self._default_value
			BLProp = bpy.props.BoolVectorProperty
			kwargs_prop |= {'size': len(typ.get_args(AttrType))}

		## Vector Int
		elif typ.get_origin(AttrType) is tuple and all(
			T is int for T in typ.get_args(AttrType)
		):
			default_value = self._default_value
			BLProp = bpy.props.IntVectorProperty
			_add_min_max_kwargs()
			kwargs_prop |= {'size': len(typ.get_args(AttrType))}

		## Vector Float
		elif typ.get_origin(AttrType) is tuple and all(
			T is float for T in typ.get_args(AttrType)
		):
			default_value = self._default_value
			BLProp = bpy.props.FloatVectorProperty
			_add_min_max_kwargs()
			_add_float_kwargs()
			kwargs_prop |= {'size': len(typ.get_args(AttrType))}

		## Generic String
		elif AttrType is str:
			default_value = self._default_value
			BLProp = bpy.props.StringProperty
			if self._str_secret:
				kwargs_prop |= {'subtype': 'PASSWORD'}
				kwargs_prop['options'].add('SKIP_SAVE')

			if self._str_cb is not None:
				kwargs_prop |= {
					'search': lambda _self, context, edit_text: self._safe_str_cb(
						_self, context, edit_text
					)
				}

		## Path
		elif AttrType is Path:
			if self._path_type is None:
				msg = 'Path BLField must define "path_type"'
				raise ValueError(msg)

			default_value = self._default_value
			BLProp = bpy.props.StringProperty
			kwargs_prop |= {
				'subtype': 'FILE_PATH' if self._path_type == 'file' else 'DIR_PATH'
			}

		## StrEnum
		elif issubclass(AttrType, enum.StrEnum):
			default_value = self._default_value
			BLProp = bpy.props.EnumProperty
			kwargs_prop |= {
				'items': [
					(
						str(value),
						AttrType.to_name(value),
						AttrType.to_name(value),  ## TODO: From AttrType.__doc__
						AttrType.to_icon(value),
						i if not self._enum_many else 2**i,
					)
					for i, value in enumerate(list(AttrType))
				]
			}
			if self._enum_many:
				kwargs_prop['options'].add('ENUM_FLAG')

		## Dynamic Enum
		elif AttrType is enum.Enum and self._enum_cb is not None:
			if self._default_value is not None:
				msg = 'When using dynamic enum, default value must be None'
				raise ValueError(msg)
			no_default_value = True

			BLProp = bpy.props.EnumProperty
			kwargs_prop |= {
				'items': lambda _self, context: self._safe_enum_cb(_self, context),
			}
			if self._enum_many:
				kwargs_prop['options'].add('ENUM_FLAG')

		## BL Reference
		elif AttrType in typ.get_args(ct.BLIDStruct):
			default_value = self._default_value
			BLProp = bpy.props.PointerProperty

		## Serializable Object
		else:
			default_value = serialize.encode(self._default_value).decode('utf-8')
			BLProp = bpy.props.StringProperty
			prop_is_serialized = True

		# Set Default Value (probably)
		if not no_default_value:
			kwargs_prop |= {'default': default_value}

		# Set Blender Property on Class __annotations__
		owner.set_prop(
			bl_attr_name,
			BLProp,
			# Update Callback Options
			no_update=not self._use_prop_update,
			update_with_name=attr_name,
			# Property Options
			name=('[JSON] ' if prop_is_serialized else '') + f'BLField: {attr_name}',
			**kwargs_prop,
		)  ## TODO: Mine description from owner class __doc__

		# Define Property Getter
		if prop_is_serialized:

			def getter(_self: BLInstance) -> AttrType:
				return serialize.decode(AttrType, getattr(_self, bl_attr_name))
		else:

			def getter(_self: BLInstance) -> AttrType:
				return getattr(_self, bl_attr_name)

		# Define Property Setter
		if prop_is_serialized:

			def setter(_self: BLInstance, value: AttrType) -> None:
				encoded_value = serialize.encode(value).decode('utf-8')
				setattr(_self, bl_attr_name, encoded_value)
		else:

			def setter(_self: BLInstance, value: AttrType) -> None:
				setattr(_self, bl_attr_name, value)

		# Initialize CachedBLProperty w/Getter and Setter
		## This is the usual descriptor assignment procedure.
		self._cached_bl_property = CachedBLProperty(getter_method=getter, persist=False)
		self._cached_bl_property.__set_name__(owner, name)
		self._cached_bl_property.setter(setter)

	def __get__(
		self, bl_instance: BLInstance | None, owner: type[BLInstance]
	) -> typ.Any:
		return self._cached_bl_property.__get__(bl_instance, owner)

	def __set__(self, bl_instance: BLInstance | None, value: typ.Any) -> None:
		if value == Signal.ResetEnumItems:
			old_items = self._safe_enum_cb(bl_instance, None)
			current_items = self._enum_cb(bl_instance, None)

			# Only Change if Changes Need Making
			if old_items != current_items:
				# Set Enum to First Item
				## Prevents the seemingly "missing" enum element bug.
				## -> Caused by the old int still trying to hang on after.
				## -> We can mitigate this by preemptively setting the enum.
				## -> Infinite recursion if we don't check current value.
				## -> May cause a hiccup (chains will trigger twice)
				## To work, there **must** be a guaranteed-available string at 0,0.
				first_old_value = old_items[0][0]
				current_value = self._cached_bl_property.__get__(
					bl_instance, bl_instance.__class__
				)
				if current_value != first_old_value:
					self._cached_bl_property.__set__(bl_instance, first_old_value)

				# Pop the Cached Enum Items
				## The next time Blender asks for the enum items, it'll update.
				self._enum_cb_cache.pop(bl_instance.instance_id, None)

				# Invalidate the Getter Cache
				## The next time the user runs __get__, they'll get the new value.
				self._cached_bl_property.__set__(bl_instance, Signal.InvalidateCache)

		elif value == Signal.ResetStrSearch:
			old_items = self._safe_str_cb(bl_instance, None)
			current_items = self._str_cb(bl_instance, None)

			# Only Change if Changes Need Making
			if old_items != current_items:
				# Set String to ''
				## Prevents the presence of an invalid value not in the new search.
				## -> Infinite recursion if we don't check current value for ''.
				## -> May cause a hiccup (chains will trigger twice)
				current_value = self._cached_bl_property.__get__(
					bl_instance, bl_instance.__class__
				)
				if current_value != '':
					self._cached_bl_property.__set__(bl_instance, '')

				# Pop the Cached String Search Items
				## The next time Blender does a str search, it'll update.
				self._str_cb_cache.pop(bl_instance.instance_id, None)

		else:
			self._cached_bl_property.__set__(bl_instance, value)
