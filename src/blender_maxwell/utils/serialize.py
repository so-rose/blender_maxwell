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

"""A fast, robust `msgspec`-based serialization tool providing for string-based persistance of many objects.

Blender provides for strong persistence guarantees based on its `bpy.types.Property` system.
In essence, properties are defined on instances of objects like nodes and sockets, and in turn, these properties can help ex. drive update chains, are **persisted on save**, and more.

The problem is fundamentally one of type support: Only "natural" types like `bool` `int`, `float`, `str`, and particular variants thereof are supported, with notable inclusions in the form of 1D/2D vectors and internal Blender pointers (which _are also persisted_).
While this forms the extent of UI support, we do extremely often want to persist things that _aren't_ one of these blessed types: At the very least things like sympy types, immutable `pydantic` models, Tidy3D objects, and more.
We want these "special" types to have the same guarantees as Blender's builtin properties.

This brings us to the intersection of `msgspec`, `pydantic`, and `bpy.props.StringProperty`.

- With a string-based property, we can "simply" serialize whatever object we want to persist, then later deserialize it into its original form.
- The property access pattern is often in a hot-loop where that same property is accessed, so even with a cache layer above it, we cannot afford to wait ex. `100ms` per a one-way (de)serialization operation.
- `pydantic` (especially V2) provides a very sane story for _many_ models, but fails in edge cases: It is simply too flexible, providing few robustness and completeness guarantees when it comes to generic serialization, while also demanding complete conformance to its `BaseModel` schema to do anything at all, which does nothing to cover the use-case of transparent type-driven serialization. To boot, its speed is fundamentally great, but still sometimes lacking in nuanced ways.
- Conversely, `msgspec` is a far, far simpler approach to _best-in-class_, type-driven serialization of _almost_ all natural Python types. It has several very important caveats, but it also supports defining custom encoding/decoding as fallbacks to normal operation.

Therefore, this module provides custom wrappers on top of `msgspec`, which are tailored to the use of several common types, including specially-enabled `pydantic` models and any `tidy3d` model encapsulated by its `dict(*)`.
We standardize on `json`, which can be easily inserted into an internal `bpy.props.StringProperty` for persistance, with access times very low.

What else did we consider?

- Direct: We tried, quite thoroughly, to keep serialization of arbitrary objects a specialized use case. The result was thousands of lines of unbearably slow, error-prone boilerplate with severe limitations.
- `json`: The standard-libary module `json` is rather inflexible, far too slow for our use case, and has no mechanisms for hooking custom objects into it.
- Use of `MsgPack`: Unfortunately, while `bpy.props.StringProperty` does have a "bytes" mode, it refuses to encode arbitrary non-UTF8 bytes. Therefore, the formal binary `MsgPack` format is out of the question, though it is preferrable in almost every other context due to both density, speed, and flexibility.

Attributes:
	NaiveEncodableType: See <https://jcristharif.com/msgspec/supported-types.html> for details.
"""

import dataclasses
import datetime as dt
import decimal
import enum
import functools
import json
import typing as typ
import uuid

import msgspec
import numpy as np
import sympy as sp
import tidy3d as td

from . import logger
from . import sympy_extra as spux

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
	| typ.Optional[typ.Any]  # noqa: UP007
	| typ.Union[typ.Any, ...]  # noqa: UP007
	| typ.NewType
	| typ.TypeAlias
	## SUPPORT:
	# | typ.Generic[typ.Any]
	# | typ.TypeVar
	# | typ.Final
	| msgspec.Raw
	## NO SUPPORT:
	# | msgspec.UNSET
)
_NaivelyEncodableTypeSet = frozenset(typ.get_args(NaivelyEncodableType))
## TODO: Use for runtime type check? Beartype?


class TypeID(enum.StrEnum):
	Complex: str = '!type=complex'
	SympyType: str = '!type=sympytype'
	SympyExpr: str = '!type=sympyexpr'
	SocketDef: str = '!type=socketdef'
	SimSymbol: str = '!type=simsymbol'
	ManagedObj: str = '!type=managedobj'
	Tidy3DObj: str = '!type=tidy3dobj'


NaiveRepresentation: typ.TypeAlias = list[TypeID, str | None, typ.Any]


def is_representation(obj: NaivelyEncodableType) -> bool:
	return isinstance(obj, list) and len(obj) == 3 and obj[0] in set(TypeID)  # noqa: PLR2004


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

	if isinstance(obj, td.components.base.Tidy3dBaseModel):
		return ['!type=tidy3dobj', None, obj._json()]

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
		return sp.sympify(obj_value).subs(spux.UNIT_BY_SYMBOL)

	if (
		issubclass(_type, td.components.base.Tidy3dBaseModel)
		and is_representation(obj)
		and obj[0] == TypeID.Tidy3DObj
	):
		obj_json = obj[2]
		return _type.parse_obj(json.loads(obj_json))

	if hasattr(_type, 'parse_as_msgspec') and (
		is_representation(obj)
		and obj[0] in [TypeID.SocketDef, TypeID.ManagedObj, TypeID.SimSymbol]
	):
		return _type.parse_as_msgspec(obj)

	msg = f'can\'t decode "{obj}" to type {type(obj)}'
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
