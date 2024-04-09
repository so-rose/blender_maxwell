import pydantic as pyd

from ... import contracts as ct
from .. import base


class MaxwellStructureBLSocket(base.MaxwellSimSocket):
	socket_type = ct.SocketType.MaxwellStructure
	bl_label = 'Maxwell Structure'


####################
# - Socket Configuration
####################
class MaxwellStructureSocketDef(pyd.BaseModel):
	socket_type: ct.SocketType = ct.SocketType.MaxwellStructure

	is_list: bool = False

	def init(self, bl_socket: MaxwellStructureBLSocket) -> None:
		if self.is_list:
			bl_socket.active_kind = ct.DataValueArray


####################
# - Blender Registration
####################
BL_REGISTER = [
	MaxwellStructureBLSocket,
]
