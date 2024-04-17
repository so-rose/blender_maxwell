
from ... import contracts as ct
from .. import base


class MaxwellSourceBLSocket(base.MaxwellSimSocket):
	socket_type = ct.SocketType.MaxwellSource
	bl_label = 'Maxwell Source'


####################
# - Socket Configuration
####################
class MaxwellSourceSocketDef(base.SocketDef):
	socket_type: ct.SocketType = ct.SocketType.MaxwellSource

	is_list: bool = False

	def init(self, bl_socket: MaxwellSourceBLSocket) -> None:
		if self.is_list:
			bl_socket.active_kind = ct.FlowKind.Array


####################
# - Blender Registration
####################
BL_REGISTER = [
	MaxwellSourceBLSocket,
]
