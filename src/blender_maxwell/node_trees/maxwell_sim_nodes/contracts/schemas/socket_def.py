import typing as typ

import bpy

from ..socket_types import SocketType


@typ.runtime_checkable
class SocketDef(typ.Protocol):
	socket_type: SocketType

	def init(self, bl_socket: bpy.types.NodeSocket) -> None: ...
