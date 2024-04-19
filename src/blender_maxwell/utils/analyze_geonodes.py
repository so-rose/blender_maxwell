import typing as typ

import bpy

INVALID_BL_SOCKET_TYPES = {
	'NodeSocketGeometry',
}


def interface(
	geonodes: bpy.types.GeometryNodeTree,  ## TODO: bpy type
	direc: typ.Literal['INPUT', 'OUTPUT'],
):
	"""Returns 'valid' GeoNodes interface sockets.

	- The Blender socket type is not something invalid (ex. "Geometry").
	- The socket has a default value.
	- The socket's direction (input/output) matches the requested direction.
	"""
	return {
		interface_item_name: bl_interface_socket
		for interface_item_name, bl_interface_socket in (
			geonodes.interface.items_tree.items()
		)
		if (
			bl_interface_socket.socket_type not in INVALID_BL_SOCKET_TYPES
			and hasattr(bl_interface_socket, 'default_value')
			and bl_interface_socket.in_out == direc
		)
	}
