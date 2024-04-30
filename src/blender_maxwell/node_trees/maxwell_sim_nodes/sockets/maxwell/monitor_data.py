from ... import contracts as ct
from .. import base


class MaxwellMonitorDataBLSocket(base.MaxwellSimSocket):
	socket_type = ct.SocketType.MaxwellMonitorData
	bl_label = 'Maxwell Monitor Data'


####################
# - Socket Configuration
####################
class MaxwellMonitorDataSocketDef(base.SocketDef):
	socket_type: ct.SocketType = ct.SocketType.MaxwellMonitorData

	def init(self, bl_socket: MaxwellMonitorDataBLSocket) -> None:
		pass


####################
# - Blender Registration
####################
BL_REGISTER = [
	MaxwellMonitorDataBLSocket,
]
