import typing as typ

import bpy
import pydantic as pyd
import tidy3d as td

from .. import base
from ... import contracts as ct


class MaxwellFDTDSimDataBLSocket(base.MaxwellSimSocket):
	socket_type = ct.SocketType.MaxwellFDTDSimData
	bl_label = 'Maxwell FDTD Simulation'

	@property
	def value(self):
		return None


####################
# - Socket Configuration
####################
class MaxwellFDTDSimDataSocketDef(pyd.BaseModel):
	socket_type: ct.SocketType = ct.SocketType.MaxwellFDTDSimData

	def init(self, bl_socket: MaxwellFDTDSimDataBLSocket) -> None:
		pass


####################
# - Blender Registration
####################
BL_REGISTER = [
	MaxwellFDTDSimDataBLSocket,
]
