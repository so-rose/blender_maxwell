import typing as typ

import pydantic as pyd

from .. import base
from ... import contracts as ct

####################
# - Blender Socket
####################
class Complex2DVectorBLSocket(base.MaxwellSimSocket):
	socket_type = ct.SocketType.Complex2DVector
	bl_label = "Complex 2D Vector"

####################
# - Socket Configuration
####################
class Complex2DVectorSocketDef(pyd.BaseModel):
	socket_type: ct.SocketType = ct.SocketType.Complex2DVector
	
	def init(self, bl_socket: Complex2DVectorBLSocket) -> None:
		pass

####################
# - Blender Registration
####################
BL_REGISTER = [
	Complex2DVectorBLSocket,
]
