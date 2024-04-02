import pydantic as pyd

from ... import contracts as ct
from .. import base


####################
# - Blender Socket
####################
class AnyBLSocket(base.MaxwellSimSocket):
	socket_type = ct.SocketType.Any
	bl_label = 'Any'


####################
# - Socket Configuration
####################
class AnySocketDef(pyd.BaseModel):
	socket_type: ct.SocketType = ct.SocketType.Any

	def init(self, bl_socket: AnyBLSocket) -> None:
		pass


####################
# - Blender Registration
####################
BL_REGISTER = [
	AnyBLSocket,
]
