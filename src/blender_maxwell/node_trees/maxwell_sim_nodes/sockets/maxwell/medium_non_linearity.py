import pydantic as pyd

from ... import contracts as ct
from .. import base


class MaxwellMediumNonLinearityBLSocket(base.MaxwellSimSocket):
	socket_type = ct.SocketType.MaxwellMediumNonLinearity
	bl_label = 'Medium Non-Linearity'


####################
# - Socket Configuration
####################
class MaxwellMediumNonLinearitySocketDef(base.SocketDef):
	socket_type: ct.SocketType = ct.SocketType.MaxwellMediumNonLinearity

	def init(self, bl_socket: MaxwellMediumNonLinearityBLSocket) -> None:
		pass


####################
# - Blender Registration
####################
BL_REGISTER = [
	MaxwellMediumNonLinearityBLSocket,
]
