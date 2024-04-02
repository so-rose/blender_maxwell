import types
import typing as typ

from .. import contracts as ct


def scan_for_socket_defs(
	sockets_module: types.ModuleType,
) -> dict[ct.SocketType, typ.Type[ct.schemas.SocketDef]]:
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
