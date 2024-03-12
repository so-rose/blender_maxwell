import typing as typ

import bpy
import sympy.physics.units as spu
import pydantic as pyd
import tidy3d as td
import scipy as sc

from .. import base
from ... import contracts as ct

VAC_SPEED_OF_LIGHT = (
	sc.constants.speed_of_light
	* spu.meter/spu.second
)

class MaxwellMonitorBLSocket(base.MaxwellSimSocket):
	socket_type = ct.SocketType.MaxwellMonitor
	bl_label = "Maxwell Monitor"

####################
# - Socket Configuration
####################
class MaxwellMonitorSocketDef(pyd.BaseModel):
	socket_type: ct.SocketType = ct.SocketType.MaxwellMonitor
	
	def init(self, bl_socket: MaxwellMonitorBLSocket) -> None:
		pass

####################
# - Blender Registration
####################
BL_REGISTER = [
	MaxwellMonitorBLSocket,
]
