import typing as typ

import bpy
import pydantic as pyd
import tidy3d as td

from .. import base
from ... import contracts as ct

class MaxwellSourceBLSocket(base.MaxwellSimSocket):
	socket_type = ct.SocketType.MaxwellSource
	bl_label = "Maxwell Source"

####################
# - Socket Configuration
####################
class MaxwellSourceSocketDef(pyd.BaseModel):
	socket_type: ct.SocketType = ct.SocketType.MaxwellSource
	
	def init(self, bl_socket: MaxwellSourceBLSocket) -> None:
		pass

####################
# - Blender Registration
####################
BL_REGISTER = [
	MaxwellSourceBLSocket,
]
