"""Tools for translating between BLMaxwell sockets and pure Blender sockets.

Attributes:
	BL_SOCKET_3D_TYPE_PREFIXES: Blender socket prefixes which indicate that the Blender socket has three values.
	BL_SOCKET_4D_TYPE_PREFIXES: Blender socket prefixes which indicate that the Blender socket has four values.
"""

import functools
import typing as typ

import bpy
import sympy as sp

from blender_maxwell.utils import extra_sympy_units as spux
from blender_maxwell.utils import logger as _logger
from . import contracts as ct
from . import sockets

log = _logger.get(__name__)

BLSocketType: typ.TypeAlias = str  ## A Blender-Defined Socket Type
BLSocketValue: typ.TypeAlias = typ.Any  ## A Blender Socket Value
BLSocketSize: typ.TypeAlias = int
DescType: typ.TypeAlias = str
Unit: typ.TypeAlias = typ.Any  ## Type of a valid unit
## TODO: Move this kind of thing to contracts


####################
# - BL Socket Size Parser
####################
BL_SOCKET_3D_TYPE_PREFIXES = {
	'NodeSocketVector',
	'NodeSocketRotation',
}
BL_SOCKET_4D_TYPE_PREFIXES = {
	'NodeSocketColor',
}


@functools.lru_cache(maxsize=4096)
def _size_from_bl_socket(
	description: str,
	bl_socket_type: BLSocketType,
):
	"""Parses the number of elements contained in a Blender interface socket.

	Since there are no 2D sockets in Blender, the user can specify "2D" in the Blender socket's description to "promise" that only the first two values will be used.
	When this is done, the third value is just never altered by the addon.

	A hard-coded set of NodeSocket<Type> prefixes are used to determine which interface sockets are, in fact, 3D.
	- For 3D sockets, a hard-coded list of Blender node socket types is used.
	- Else, it is a 1D socket type.
	"""
	if description.startswith('2D'):
		return 2
	if any(
		bl_socket_type.startswith(bl_socket_3d_type_prefix)
		for bl_socket_3d_type_prefix in BL_SOCKET_3D_TYPE_PREFIXES
	):
		return 3
	if any(
		bl_socket_type.startswith(bl_socket_4d_type_prefix)
		for bl_socket_4d_type_prefix in BL_SOCKET_4D_TYPE_PREFIXES
	):
		return 4

	return 1


####################
# - BL Socket Type / Unit Parser
####################
@functools.lru_cache(maxsize=4096)
def _socket_type_from_bl_socket(
	description: str,
	bl_socket_type: BLSocketType,
) -> ct.SocketType:
	"""Parse a Blender socket for a matching BLMaxwell socket type, relying on both the Blender socket type and user-generated hints in the description.

	Arguments:
		description: The description from Blender socket, aka. `bl_socket.description`.
		bl_socket_type: The Blender socket type, aka. `bl_socket.socket_type`.

	Returns:
		The type of a MaxwellSimSocket that corresponds to the Blender socket.
	"""
	size = _size_from_bl_socket(description, bl_socket_type)

	# Determine Socket Type Directly
	## The naive mapping from BL socket -> Maxwell socket may be good enough.
	if (
		direct_socket_type := ct.BL_SOCKET_DIRECT_TYPE_MAP.get((bl_socket_type, size))
	) is None:
		msg = "Blender interface socket has no mapping among 'MaxwellSimSocket's."
		raise ValueError(msg)

	# (No Description) Return Direct Socket Type
	if ct.BL_SOCKET_DESCR_ANNOT_STRING not in description:
		return direct_socket_type

	# Parse Description for Socket Type
	## The "2D" token is special; don't include it if it's there.
	descr_params = description.split(ct.BL_SOCKET_DESCR_ANNOT_STRING)[0]
	directive = (
		_tokens[0] if (_tokens := descr_params.split(' '))[0] != '2D' else _tokens[1]
	)
	if directive == 'Preview':
		return direct_socket_type  ## TODO: Preview element handling

	if (
		socket_type := ct.BL_SOCKET_DESCR_TYPE_MAP.get(
			(directive, bl_socket_type, size)
		)
	) is None:
		msg = f'Socket description "{(directive, bl_socket_type, size)}" doesn\'t map to a socket type + unit'
		raise ValueError(msg)

	return socket_type


####################
# - BL Socket Interface Definition
####################
@functools.lru_cache(maxsize=4096)
def _socket_def_from_bl_socket(
	description: str,
	bl_socket_type: BLSocketType,
) -> ct.SocketType:
	return sockets.SOCKET_DEFS[_socket_type_from_bl_socket(description, bl_socket_type)]


def socket_def_from_bl_socket(
	bl_interface_socket: bpy.types.NodeTreeInterfaceSocket,
) -> sockets.base.SocketDef:
	"""Computes an appropriate (no-arg) SocketDef from the given `bl_interface_socket`, by parsing it."""
	return _socket_def_from_bl_socket(
		bl_interface_socket.description, bl_interface_socket.bl_socket_idname
	)


####################
# - Extract Default Interface Socket Value
####################
def _read_bl_socket_default_value(
	description: str,
	bl_socket_type: BLSocketType,
	bl_socket_value: BLSocketValue,
	unit_system: dict | None = None,
	allow_unit_not_in_unit_system: bool = False,
) -> typ.Any:
	# Parse the BL Socket Type and Value
	## The 'lambda' delays construction until size is determined.
	socket_type = _socket_type_from_bl_socket(description, bl_socket_type)
	parsed_socket_value = {
		1: lambda: bl_socket_value,
		2: lambda: sp.Matrix(tuple(bl_socket_value)[:2]),
		3: lambda: sp.Matrix(tuple(bl_socket_value)),
		4: lambda: sp.Matrix(tuple(bl_socket_value)),
	}[_size_from_bl_socket(description, bl_socket_type)]()

	# Add Unit-System Unit to Parsed
	## Use the matching socket type to lookup the unit in the unit system.
	if unit_system is not None:
		if (unit := unit_system.get(socket_type)) is None:
			if allow_unit_not_in_unit_system:
				return parsed_socket_value

			msg = f'Unit system does not provide a unit for {socket_type}'
			raise RuntimeError(msg)

		return parsed_socket_value * unit
	return parsed_socket_value


def read_bl_socket_default_value(
	bl_interface_socket: bpy.types.NodeTreeInterfaceSocket,
	unit_system: dict | None = None,
	allow_unit_not_in_unit_system: bool = False,
) -> typ.Any:
	"""Reads the `default_value` of a Blender socket, guaranteeing a well-formed value consistent with the passed unit system.

	Arguments:
		bl_interface_socket: The Blender interface socket to analyze for description, socket type, and default value.
		unit_system: The mapping from BLMaxwell SocketType to corresponding unit, used to apply the appropriate unit to the output.

	Returns:
		The parsed, well-formed version of `bl_socket.default_value`, of the appropriate form and unit.

	"""
	return _read_bl_socket_default_value(
		bl_interface_socket.description,
		bl_interface_socket.bl_socket_idname,
		bl_interface_socket.default_value,
		unit_system=unit_system,
		allow_unit_not_in_unit_system=allow_unit_not_in_unit_system,
	)


def _writable_bl_socket_value(
	description: str,
	bl_socket_type: BLSocketType,
	value: typ.Any,
	unit_system: dict | None = None,
	allow_unit_not_in_unit_system: bool = False,
) -> typ.Any:
	socket_type = _socket_type_from_bl_socket(description, bl_socket_type)

	# Retrieve Unit-System Unit
	if unit_system is not None:
		if (unit := unit_system.get(socket_type)) is None:
			if allow_unit_not_in_unit_system:
				_bl_socket_value = value
			else:
				msg = f'Unit system does not provide a unit for {socket_type}'
				raise RuntimeError(msg)
		else:
			_bl_socket_value = spux.scale_to_unit(value, unit)
	else:
		_bl_socket_value = value

	# Compute Blender Socket Value
	if isinstance(_bl_socket_value, sp.Basic | sp.MatrixBase):
		bl_socket_value = spux.sympy_to_python(_bl_socket_value)
	else:
		bl_socket_value = _bl_socket_value

	if _size_from_bl_socket(description, bl_socket_type) == 2:  # noqa: PLR2004
		bl_socket_value = bl_socket_value[:2]
	return bl_socket_value


def writable_bl_socket_value(
	bl_interface_socket: bpy.types.NodeTreeInterfaceSocket,
	value: typ.Any,
	unit_system: dict | None = None,
	allow_unit_not_in_unit_system: bool = False,
) -> typ.Any:
	"""Processes a value to be ready-to-write to a Blender socket.

	Arguments:
		bl_interface_socket: The Blender interface socket to analyze
		value: The value to prepare for writing to the given Blender socket.
		unit_system: The mapping from BLMaxwell SocketType to corresponding unit, used to scale the value to the the appropriate unit.

	Returns:
		A value corresponding to the input, which is guaranteed to be compatible with the Blender socket (incl. via a GeoNodes modifier), as well as correctly scaled with respect to the given unit system.
	"""
	return _writable_bl_socket_value(
		bl_interface_socket.description,
		bl_interface_socket.bl_socket_idname,
		value,
		unit_system=unit_system,
		allow_unit_not_in_unit_system=allow_unit_not_in_unit_system,
	)
