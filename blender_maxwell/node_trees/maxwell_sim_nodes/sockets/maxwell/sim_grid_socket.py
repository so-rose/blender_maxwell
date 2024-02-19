import typing as typ

import bpy
import pydantic as pyd
import tidy3d as td

from .. import base
from ... import contracts

class MaxwellSimGridBLSocket(base.BLSocket):
	socket_type = contracts.SocketType.MaxwellSimGrid
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
class MaxwellSimGridSocketDef(pyd.BaseModel):
	socket_type: contracts.SocketType = contracts.SocketType.MaxwellSimGrid
	label: str
	
	def init(self, bl_socket: MaxwellSimGridBLSocket) -> None:
		pass

####################
# - Blender Registration
####################
BL_REGISTER = [
	MaxwellSimGridBLSocket,
]
