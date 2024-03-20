import typing as typ

import bpy
import pydantic as pyd
import tidy3d as td

from .. import base
from ... import contracts as ct


class MaxwellSimGridAxisBLSocket(base.MaxwellSimSocket):
	socket_type = ct.SocketType.MaxwellSimGridAxis
	bl_label = 'Maxwell Bound Box'


####################
# - Socket Configuration
####################
class MaxwellSimGridAxisSocketDef(pyd.BaseModel):
	socket_type: ct.SocketType = ct.SocketType.MaxwellSimGridAxis

	def init(self, bl_socket: MaxwellSimGridAxisBLSocket) -> None:
		pass


####################
# - Blender Registration
####################
BL_REGISTER = [
	MaxwellSimGridAxisBLSocket,
]
