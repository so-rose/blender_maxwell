
import pydantic as pyd

from ... import contracts as ct
from .. import base


class MaxwellFDTDSimBLSocket(base.MaxwellSimSocket):
	socket_type = ct.SocketType.MaxwellFDTDSim
	bl_label = 'Maxwell FDTD Simulation'

	@property
	def value(self) -> None:
		return None


####################
# - Socket Configuration
####################
class MaxwellFDTDSimSocketDef(pyd.BaseModel):
	socket_type: ct.SocketType = ct.SocketType.MaxwellFDTDSim

	def init(self, bl_socket: MaxwellFDTDSimBLSocket) -> None:
		pass


####################
# - Blender Registration
####################
BL_REGISTER = [
	MaxwellFDTDSimBLSocket,
]
