import typing as typ
import typing_extensions as typx

import pydantic as pyd
import sympy as sp
import sympy.physics.units as spu

import bpy

from ...utils import extra_sympy_units as spuex
from . import contracts as ct
from .contracts import SocketType as ST
from . import sockets as sck

# TODO: Caching?
# TODO: Move the manual labor stuff to contracts

BLSocketType = str  ## A Blender-Defined Socket Type
BLSocketSize = int
DescType = str
Unit = typ.Any  ## Type of a valid unit

####################
# - Socket to SocketDef
####################
SOCKET_DEFS = {
	socket_type: getattr(
		sck,
		socket_type.value.removesuffix("SocketType") + "SocketDef",
	)
	for socket_type in ST
	if hasattr(
		sck,
		socket_type.value.removesuffix("SocketType") + "SocketDef"
	)
}
## TODO: Bit of a hack. Is it robust enough?

for socket_type in ST:
	if not hasattr(
		sck,
		socket_type.value.removesuffix("SocketType") + "SocketDef",
	):
		print("Missing SocketDef for", socket_type.value)


####################
# - BL Socket Size Parser
####################
BL_SOCKET_3D_TYPE_PREFIXES = {
	"NodeSocketVector",
	"NodeSocketRotation",
}
BL_SOCKET_4D_TYPE_PREFIXES = {
	"NodeSocketColor",
}
def size_from_bl_interface_socket(
	bl_interface_socket: bpy.types.NodeTreeInterfaceSocket
) -> typx.Literal[1, 2, 3, 4]:
	"""Parses the `size`, aka. number of elements, contained within the `default_value` of a Blender interface socket.
	
	Since there are no 2D sockets in Blender, the user can specify "2D" in the Blender socket's description to "promise" that only the first two values will be used.
	When this is done, the third value is left entirely untouched by this entire system.
	
	A hard-coded set of NodeSocket<Type> prefixes are used to determine which interface sockets are, in fact, 3D.
	- For 3D sockets, a hard-coded list of Blender node socket types is used.
	- Else, it is a 1D socket type.
	"""
	if bl_interface_socket.description.startswith("2D"): return 2
	if any(
		bl_interface_socket.socket_type.startswith(bl_socket_3d_type_prefix)
		for bl_socket_3d_type_prefix in BL_SOCKET_3D_TYPE_PREFIXES
	):
		return 3
	if any(
		bl_interface_socket.socket_type.startswith(bl_socket_4d_type_prefix)
		for bl_socket_4d_type_prefix in BL_SOCKET_4D_TYPE_PREFIXES
	):
		return 4
	
	return 1


####################
# - BL Socket Type / Unit Parser
####################
def parse_bl_interface_socket(
	bl_interface_socket: bpy.types.NodeTreeInterfaceSocket,
) -> tuple[ST, sp.Expr | None]:
	"""Parse a Blender interface socket by parsing its description, falling back to any direct type links.
	
	Arguments:
		bl_interface_socket: An interface socket associated with the global input to a node tree.
	
	Returns:
		The type of a corresponding MaxwellSimSocket, as well as a unit (if a particular unit was requested by the Blender interface socket).
	"""
	size = size_from_bl_interface_socket(bl_interface_socket)
	
	# Determine Direct Socket Type
	if (
		direct_socket_type := ct.BL_SOCKET_DIRECT_TYPE_MAP.get(
			(bl_interface_socket.socket_type, size)
		)
	) is None:
		msg = "Blender interface socket has no mapping among 'MaxwellSimSocket's."
		raise ValueError(msg)
	
	# (Maybe) Return Direct Socket Type
	## When there's no description, that's it; return.
	if not ct.BL_SOCKET_DESCR_ANNOT_STRING in bl_interface_socket.description:
		return (direct_socket_type, None)
	
	# Parse Description for Socket Type
	tokens = (
		_tokens
		if (_tokens := bl_interface_socket.description.split(" "))[0] != "2D"
		else _tokens[1:]
	) ## Don't include the "2D" token, if defined.
	if (
		socket_type := ct.BL_SOCKET_DESCR_TYPE_MAP.get(
			(tokens[0], bl_interface_socket.socket_type, size)
		)
	) is None:
		return (direct_socket_type, None)  ## Description doesn't map to anything
	
	# Determine Socket Unit (to use instead of "unit system")
	## This is entirely OPTIONAL
	socket_unit = None
	if socket_type in ct.SOCKET_UNITS:
		## Case: Unit is User-Defined
		if len(tokens) > 1 and "(" in tokens[1] and ")" in tokens[1]:
			# Compute (<unit_str>) as Unit Token
			unit_token = tokens[1].removeprefix("(").removesuffix(")")
			
			# Compare Unit Token to Valid Sympy-Printed Units
			socket_unit = _socket_unit if (_socket_unit := [
				unit
				for unit in ct.SOCKET_UNITS[socket_type]["values"].values()
				if str(unit) == unit_token
			]) else ct.SOCKET_UNITS[socket_type]["values"][
				ct.SOCKET_UNITS[socket_type]["default"]
			]
			## TODO: Enforce abbreviated sympy printing here, not globally
	
	return (socket_type, socket_unit)


####################
# - BL Socket Interface Definition
####################
def socket_def_from_bl_interface_socket(
	bl_interface_socket: bpy.types.NodeTreeInterfaceSocket,
):
	"""Computes an appropriate (no-arg) SocketDef from the given `bl_interface_socket`, by parsing it.
	"""
	return SOCKET_DEFS[
		parse_bl_interface_socket(bl_interface_socket)[0]
	]


####################
# - Extract Default Interface Socket Value
####################
def value_from_bl(
	bl_interface_socket: bpy.types.NodeTreeInterfaceSocket,
	unit_system: dict | None = None,
) -> typ.Any:
	"""Reads the value of any Blender socket, and writes its `default_value` to the `value` of any `MaxwellSimSocket`.
	- If the size of the Blender socket is >1, then `value` is written to as a `sympy.Matrix`.
	- If a unit system is given, then the Blender socket is matched to a `MaxwellSimSocket`, which is used to lookup an appropriate unit in the given `unit_system`.
	
	"""
	## TODO: Consider sympy.S()'ing the default_value
	parsed_bl_socket_value = {
		1: lambda: bl_interface_socket.default_value,
		2: lambda: sp.Matrix(tuple(bl_interface_socket.default_value)[:2]),
		3: lambda: sp.Matrix(tuple(bl_interface_socket.default_value)),
		4: lambda: sp.Matrix(tuple(bl_interface_socket.default_value)),
	}[size_from_bl_interface_socket(bl_interface_socket)]()
	## The 'lambda' delays construction until size is determined
	
	socket_type, unit = parse_bl_interface_socket(bl_interface_socket)
	
	# Add Unit to Parsed (if relevant)
	if unit is not None:
		parsed_bl_socket_value *= unit
	elif unit_system is not None:
		parsed_bl_socket_value *= unit_system[socket_type]
	
	return parsed_bl_socket_value

####################
# - Convert to Blender-Compatible Value
####################
def make_scalar_bl_compat(scalar: typ.Any) -> typ.Any:
	"""Blender doesn't accept ex. Sympy numbers as values.
	Therefore, we need to do some conforming.
	
	Currently hard-coded; this is probably best.
	"""
	if isinstance(scalar, sp.Integer):
		return int(scalar)
	elif isinstance(scalar, sp.Float):
		return float(scalar)
	elif isinstance(scalar, sp.Rational):
		return float(scalar)
	elif isinstance(scalar, sp.Expr):
		return float(scalar.n())
	## TODO: More?
	
	return scalar

def value_to_bl(
	bl_interface_socket: bpy.types.NodeSocket,
	value: typ.Any,
	unit_system: dict | None = None,
) -> typ.Any:
	socket_type, unit = parse_bl_interface_socket(bl_interface_socket)
	
	# Set Socket
	if unit is not None:
		bl_socket_value = spu.convert_to(value, unit) / unit
	elif (
		unit_system is not None
		and socket_type in unit_system
	):
		bl_socket_value = spu.convert_to(
			value, unit_system[socket_type]
		) / unit_system[socket_type]
	else:
		bl_socket_value = value
	
	return {
		1: lambda: make_scalar_bl_compat(bl_socket_value),
		2: lambda: tuple([
			make_scalar_bl_compat(bl_socket_value[0]),
			make_scalar_bl_compat(bl_socket_value[1]),
			bl_interface_socket.default_value[2]
			## Don't touch (unused) 3rd bl_socket coordinate
		]),
		3: lambda: tuple([
			make_scalar_bl_compat(el)
			for el in bl_socket_value
		]),
		4: lambda: tuple([
			make_scalar_bl_compat(el)
			for el in bl_socket_value
		]),
	}[size_from_bl_interface_socket(bl_interface_socket)]()
	## The 'lambda' delays construction until size is determined
