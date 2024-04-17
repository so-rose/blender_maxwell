
from ... import contracts as ct
from .. import base


class MaxwellStructureBLSocket(base.MaxwellSimSocket):
	socket_type = ct.SocketType.MaxwellStructure
	bl_label = 'Maxwell Structure'


####################
# - Socket Configuration
####################
class MaxwellStructureSocketDef(base.SocketDef):
	socket_type: ct.SocketType = ct.SocketType.MaxwellStructure

	is_list: bool = False

	def init(self, bl_socket: MaxwellStructureBLSocket) -> None:
		if self.is_list:
			bl_socket.active_kind = ct.FlowKind.ValueArray


####################
# - Blender Registration
####################
BL_REGISTER = [
	MaxwellStructureBLSocket,
]
