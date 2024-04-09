import pydantic as pyd

from ... import contracts as ct
from .. import base


class MaxwellMonitorBLSocket(base.MaxwellSimSocket):
	socket_type = ct.SocketType.MaxwellMonitor
	bl_label = 'Maxwell Monitor'


####################
# - Socket Configuration
####################
class MaxwellMonitorSocketDef(pyd.BaseModel):
	socket_type: ct.SocketType = ct.SocketType.MaxwellMonitor

	is_list: bool = False

	def init(self, bl_socket: MaxwellMonitorBLSocket) -> None:
		if self.is_list:
			bl_socket.active_kind = ct.DataValueArray


####################
# - Blender Registration
####################
BL_REGISTER = [
	MaxwellMonitorBLSocket,
]
