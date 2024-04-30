"""Tools for translating between BLMaxwell sockets and pure Blender sockets.

Attributes:
	BL_SOCKET_3D_TYPE_PREFIXES: Blender socket prefixes which indicate that the Blender socket has three values.
	BL_SOCKET_4D_TYPE_PREFIXES: Blender socket prefixes which indicate that the Blender socket has four values.
"""

import typing as typ

import bpy

from blender_maxwell.utils import extra_sympy_units as spux
from blender_maxwell.utils import logger as _logger

from . import contracts as ct
from . import sockets

log = _logger.get(__name__)


####################
# - Blender -> Socket Def(s)
####################
def socket_def_from_bl_isocket(
	bl_isocket: bpy.types.NodeTreeInterfaceSocket,
) -> sockets.base.SocketDef | None:
	"""Deduces and constructs an appropriate SocketDef to match the given `bl_interface_socket`."""
	blsck_info = ct.BLSocketType.info_from_bl_isocket(bl_isocket)
	if blsck_info.has_support and not blsck_info.is_preview:
		# Map Expr Socket
		## -> Accounts for any combo of shape/MathType/PhysicalType.
		if blsck_info.socket_type == ct.SocketType.Expr:
			return sockets.ExprSocketDef(
				shape=blsck_info.size.shape,
				mathtype=blsck_info.mathtype,
				physical_type=blsck_info.physical_type,
				default_unit=ct.UNITS_BLENDER[blsck_info.physical_type],
				default_value=blsck_info.default_value,
			)

		## TODO: Explicitly map default to other supported SocketDef constructors

		return sockets.SOCKET_DEFS[blsck_info.socket_type]()
	return None


def sockets_from_geonodes(
	geonodes: bpy.types.GeometryNodeTree,
) -> dict[ct.SocketName, sockets.base.SocketDef]:
	"""Deduces and constructs appropriate SocketDefs to match all input sockets to the given GeoNodes tree."""
	raw_socket_defs = {
		socket_name: socket_def_from_bl_isocket(bl_isocket)
		for socket_name, bl_isocket in geonodes.interface.items_tree.items()
	}
	return {
		socket_name: socket_def
		for socket_name, socket_def in raw_socket_defs.items()
		if socket_def is not None
	}


## TODO: Make it fast, it's in a hot loop...
def info_from_geonodes(
	geonodes: bpy.types.GeometryNodeTree,
) -> dict[ct.SocketName, ct.BLSocketInfo]:
	"""Deduces and constructs appropriate SocketDefs to match all input sockets to the given GeoNodes tree."""
	return {
		socket_name: ct.BLSocketType.info_from_bl_isocket(bl_isocket)
		for socket_name, bl_isocket in geonodes.interface.items_tree.items()
	}
