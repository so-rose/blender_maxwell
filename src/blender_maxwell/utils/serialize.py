import functools
import typing as typ

import msgspec
import sympy as sp
import sympy.physics.units as spu

from . import extra_sympy_units as spux
from . import logger

log = logger.get(__name__)

EncodableValue: typ.TypeAlias = typ.Any  ## msgspec-compatible

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
