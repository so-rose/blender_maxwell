import typing as typ

import bpy
import pydantic as pyd
import tidy3d as td

from .. import base
from ... import contracts as ct

class MaxwellFDTDSimBLSocket(base.MaxwellSimSocket):
	socket_type = ct.SocketType.MaxwellFDTDSim
	bl_label = "Maxwell FDTD Simulation"

####################
# - Socket Configuration
####################
class MaxwellFDTDSimSocketDef(pyd.BaseModel):
	socket_type: ct.SocketType = ct.SocketType.MaxwellFDTDSim
	
	def init(self, bl_socket: MaxwellFDTDSimBLSocket) -> None:
		pass

####################
# - Blender Registration
####################
BL_REGISTER = [
	MaxwellFDTDSimBLSocket,
]
