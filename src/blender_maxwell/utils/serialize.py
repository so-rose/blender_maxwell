"""Attributes:
NaiveEncodableType:
See <https://jcristharif.com/msgspec/supported-types.html> for details.
"""

import dataclasses
import datetime as dt
import decimal
import enum
import functools
import typing as typ
import uuid

import msgspec
import sympy as sp

from . import extra_sympy_units as spux
from . import logger

log = logger.get(__name__)

####################
# - Serialization Types
####################
NaivelyEncodableType: typ.TypeAlias = (
	None
	| bool
	| int
	| float
	| str
	| bytes
	| bytearray
	## NO SUPPORT:
	# | memoryview
	| tuple
	| list
	| dict
	| set
	| frozenset
	## NO SUPPORT:
	# | typ.Literal
	| typ.Collection
	## NO SUPPORT:
	# | typ.Sequence  ## -> list
	# | typ.MutableSequence  ## -> list
	# | typ.AbstractSet  ## -> set
	# | typ.MutableSet  ## -> set
	# | typ.Mapping  ## -> dict
	# | typ.MutableMapping  ## -> dict
	| typ.TypedDict
	| typ.NamedTuple
	| dt.datetime
	| dt.date
	| dt.time
	| dt.timedelta
	| uuid.UUID
	| decimal.Decimal
	## NO SUPPORT:
	# | enum.Enum
	| enum.IntEnum
	| enum.Flag
	| enum.IntFlag
	| dataclasses.dataclass
	| typ.Optional
	| typ.Union
	| typ.NewType
	| typ.TypeAlias
	| typ.TypeAliasType
	| typ.Generic
	| typ.TypeVar
	| typ.Final
	| msgspec.Raw
	## NO SUPPORT:
	# | msgspec.UNSET
)
_NaivelyEncodableTypeSet = frozenset(typ.get_args(NaivelyEncodableType))


class TypeID(enum.StrEnum):
	Complex: str = '!type=complex'
	SympyType: str = '!type=sympytype'
	SocketDef: str = '!type=socketdef'
	ManagedObj: str = '!type=managedobj'


NaiveRepresentation: typ.TypeAlias = list[TypeID, str | None, typ.Any]


def is_representation(obj: NaivelyEncodableType) -> bool:
	return isinstance(obj, list) and obj[0] in set(TypeID) and len(obj) == 3  # noqa: PLR2004


####################
# - Serialization Hooks
####################
def _enc_hook(obj: typ.Any) -> NaivelyEncodableType:
	"""Translates types not natively supported by `msgspec`, to an encodable form supported by `msgspec`.

	Parameters:
	obj: The object of arbitrary type to transform into an encodable value.

	Returns:
	A value encodable by `msgspec`.

	Raises:
	NotImplementedError: When the type transformation hasn't been implemented.
	"""
	if isinstance(obj, complex):
		return ['!type=complex', None, (obj.real, obj.imag)]

	if isinstance(obj, spux.SympyType):
		return ['!type=sympytype', None, sp.srepr(obj)]

	if hasattr(obj, 'dump_as_msgspec'):
		return obj.dump_as_msgspec()

	msg = f'Can\'t encode "{obj}" of type {type(obj)}'
	raise NotImplementedError(msg)


def _dec_hook(_type: type, obj: NaivelyEncodableType) -> typ.Any:
	"""Translates the `msgspec`-encoded form of an object back to its true form.

	Parameters:
		_type: The type to transform the `msgspec`-encoded object back into.
		obj: The naively decoded object to transform back into its actual type.

	Returns:
		A value encodable by `msgspec`.

	Raises:
		NotImplementedError: When the type transformation hasn't been implemented.
	"""
	if _type is complex or (is_representation(obj) and obj[0] == TypeID.Complex):
		obj_value = obj[2]
		return complex(obj_value[0], obj_value[1])

	if _type in typ.get_args(spux.SympyType) or (
		is_representation(obj) and obj[0] == TypeID.SympyType
	):
		obj_value = obj[2]
		return sp.sympify(obj_value).subs(spux.ALL_UNIT_SYMBOLS)

	if hasattr(obj, 'parse_as_msgspec'):
		return _type.parse_as_msgspec(obj)

	msg = f'Can\'t decode "{obj}" to type {type(obj)}'
	raise NotImplementedError(msg)


####################
# - Global Encoders / Decoders
####################
_ENCODER = msgspec.json.Encoder(enc_hook=_enc_hook, order='deterministic')


@functools.cache
def _DECODER(_type: type) -> msgspec.json.Decoder:  # noqa: N802
	"""Retrieve a suitable `msgspec.json.Decoder` by-type.

	Parameters:
		_type: The type to retrieve a decoder for.

	Returns:
		A suitable decoder.
	"""
	return msgspec.json.Decoder(type=_type, dec_hook=_dec_hook)


####################
# - Encoder / Decoder Functions
####################
def encode(obj: typ.Any) -> bytes:
	return _ENCODER.encode(obj)


def decode(_type: type, obj: str | bytes) -> typ.Any:
	return _DECODER(_type).decode(obj)
