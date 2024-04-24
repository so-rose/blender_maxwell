from ... import contracts as ct
from .. import base


class MaxwellFDTDSimDataBLSocket(base.MaxwellSimSocket):
	socket_type = ct.SocketType.MaxwellFDTDSimData
	bl_label = 'Maxwell FDTD Simulation'


####################
# - Socket Configuration
####################
class MaxwellFDTDSimDataSocketDef(base.SocketDef):
	socket_type: ct.SocketType = ct.SocketType.MaxwellFDTDSimData

	def init(self, bl_socket: MaxwellFDTDSimDataBLSocket) -> None:
		pass


####################
# - Blender Registration
####################
BL_REGISTER = [
	MaxwellFDTDSimDataBLSocket,
]
