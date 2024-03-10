import typing as typ

import pydantic as pyd

from .. import base
from ... import contracts as ct

####################
# - Blender Socket
####################
class Complex3DVectorBLSocket(base.MaxwellSimSocket):
	socket_type = ct.SocketType.Complex3DVector
	bl_label = "Complex 3D Vector"

####################
# - Socket Configuration
####################
class Complex3DVectorSocketDef(pyd.BaseModel):
	socket_type: ct.SocketType = ct.SocketType.Complex3DVector
	
	def init(self, bl_socket: Complex3DVectorBLSocket) -> None:
		pass

####################
# - Blender Registration
####################
BL_REGISTER = [
	Complex3DVectorBLSocket,
]
