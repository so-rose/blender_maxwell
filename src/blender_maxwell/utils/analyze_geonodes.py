import typing_extensions as typx

INVALID_BL_SOCKET_TYPES = {
	'NodeSocketGeometry',
}


def interface(
	geo_nodes,
	direc: typx.Literal['INPUT', 'OUTPUT'],
):
	"""Returns 'valid' GeoNodes interface sockets, meaning that:
	- The Blender socket type is not something invalid (ex. "Geometry").
	- The socket has a default value.
	- The socket's direction (input/output) matches the requested direction.
	"""
	return {
		interface_item_name: bl_interface_socket
		for interface_item_name, bl_interface_socket in (
			geo_nodes.interface.items_tree.items()
		)
		if all(
			[
				bl_interface_socket.socket_type not in INVALID_BL_SOCKET_TYPES,
				hasattr(bl_interface_socket, 'default_value'),
				bl_interface_socket.in_out == direc,
			]
		)
	}
