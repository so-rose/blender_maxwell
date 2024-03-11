import typing as typ

import bpy
import pydantic as pyd
import tidy3d as td

from .. import base
from ... import contracts as ct

class MaxwellMediumNonLinearityBLSocket(base.MaxwellSimSocket):
	socket_type = ct.SocketType.MaxwellMediumNonLinearity
	bl_label = "Medium Non-Linearity"

####################
# - Socket Configuration
####################
class MaxwellMediumNonLinearitySocketDef(pyd.BaseModel):
	socket_type: ct.SocketType = ct.SocketType.MaxwellMediumNonLinearity
	
	def init(self, bl_socket: MaxwellMediumNonLinearityBLSocket) -> None:
		pass

####################
# - Blender Registration
####################
BL_REGISTER = [
	MaxwellMediumNonLinearityBLSocket,
]
