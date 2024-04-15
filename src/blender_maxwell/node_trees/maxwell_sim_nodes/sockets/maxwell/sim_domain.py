
from ... import contracts as ct
from .. import base


class MaxwellSimDomainBLSocket(base.MaxwellSimSocket):
	socket_type = ct.SocketType.MaxwellSimDomain
	bl_label = 'Sim Domain'


####################
# - Socket Configuration
####################
class MaxwellSimDomainSocketDef(base.SocketDef):
	socket_type: ct.SocketType = ct.SocketType.MaxwellSimDomain

	def init(self, bl_socket: MaxwellSimDomainBLSocket) -> None:
		pass


####################
# - Blender Registration
####################
BL_REGISTER = [
	MaxwellSimDomainBLSocket,
]
