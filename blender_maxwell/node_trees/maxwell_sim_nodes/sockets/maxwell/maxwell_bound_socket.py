import typing as typ

import bpy
import pydantic as pyd
import tidy3d as td

from .. import base
from ... import contracts

class MaxwellBoundBLSocket(base.BLSocket):
	socket_type = contracts.SocketType.MaxwellBound
	socket_color = (0.8, 0.8, 0.4, 1.0)
	
	bl_label = "Maxwell Bound"
	
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
class MaxwellBoundSocketDef(pyd.BaseModel):
	socket_type: contracts.SocketType = contracts.SocketType.MaxwellBound
	label: str
	
	def init(self, bl_socket: MaxwellBoundBLSocket) -> None:
		pass

####################
# - Blender Registration
####################
BL_REGISTER = [
	MaxwellBoundBLSocket,
]
