
import pydantic as pyd

from ... import contracts as ct
from .. import base


class MaxwellTemporalShapeBLSocket(base.MaxwellSimSocket):
	socket_type = ct.SocketType.MaxwellTemporalShape
	bl_label = 'Maxwell Temporal Shape'


####################
# - Socket Configuration
####################
class MaxwellTemporalShapeSocketDef(pyd.BaseModel):
	socket_type: ct.SocketType = ct.SocketType.MaxwellTemporalShape

	def init(self, bl_socket: MaxwellTemporalShapeBLSocket) -> None:
		pass


####################
# - Blender Registration
####################
BL_REGISTER = [
	MaxwellTemporalShapeBLSocket,
]
