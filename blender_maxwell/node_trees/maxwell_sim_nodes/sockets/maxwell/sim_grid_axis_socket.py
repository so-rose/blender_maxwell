import typing as typ

import bpy
import pydantic as pyd
import tidy3d as td

from .. import base
from ... import contracts

class MaxwellSimGridAxisBLSocket(base.BLSocket):
	socket_type = contracts.SocketType.MaxwellSimGridAxis
	bl_label = "Maxwell Bound Box"
	
	compatible_types = {
		td.BoundarySpec: {}
	}
	
	####################
	# - Computation of Default Value
	####################
	@property
	def default_value(self) -> td.BoundarySpec:
		return td.BoundarySpec()
	
	@default_value.setter
	def default_value(self, value: typ.Any) -> None:
		return None

####################
# - Socket Configuration
####################
class MaxwellSimGridAxisSocketDef(pyd.BaseModel):
	socket_type: contracts.SocketType = contracts.SocketType.MaxwellSimGridAxis
	label: str
	
	def init(self, bl_socket: MaxwellSimGridAxisBLSocket) -> None:
		pass

####################
# - Blender Registration
####################
BL_REGISTER = [
	MaxwellSimGridAxisBLSocket,
]
