import types

from .. import contracts as ct


## TODO: Replace with BL_SOCKET_DEFS export from each module, a little like BL_NODES.
def scan_for_socket_defs(
	sockets_module: types.ModuleType,
) -> dict:
	return {
		socket_type: getattr(
			sockets_module,
			socket_type.value.removesuffix('SocketType') + 'SocketDef',
		)
		for socket_type in ct.SocketType
		if hasattr(
			sockets_module, socket_type.value.removesuffix('SocketType') + 'SocketDef'
		)
	}


## TODO: Function for globals() filling too.
