import typing as typ

import bpy
import pydantic as pyd
import tidy3d as td

from .. import base
from ... import contracts

class MaxwellBoundBoxBLSocket(base.BLSocket):
	socket_type = contracts.SocketType.MaxwellBoundBox
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
class MaxwellBoundBoxSocketDef(pyd.BaseModel):
	socket_type: contracts.SocketType = contracts.SocketType.MaxwellBoundBox
	label: str
	
	def init(self, bl_socket: MaxwellBoundBoxBLSocket) -> None:
		pass

####################
# - Blender Registration
####################
BL_REGISTER = [
	MaxwellBoundBoxBLSocket,
]
