import typing as typ

import bpy
import sympy as sp
import pydantic as pyd

from .. import base
from ... import contracts as ct

####################
# - Blender Socket
####################
class AnyBLSocket(base.MaxwellSimSocket):
	socket_type = ct.SocketType.Any
	bl_label = "Any"

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
