import typing as typ

import bpy
import pydantic as pyd
import tidy3d as td

from .. import base
from ... import contracts

class MaxwellTemporalShapeBLSocket(base.BLSocket):
	socket_type = contracts.SocketType.MaxwellTemporalShape
	bl_label = "Maxwell Temporal Shape"
	
	####################
	# - Computation of Default Value
	####################
	@property
	def default_value(self) -> td.Medium:
		return None
	
	@default_value.setter
	def default_value(self, value: typ.Any) -> None:
		pass

####################
# - Socket Configuration
####################
class MaxwellTemporalShapeSocketDef(pyd.BaseModel):
	socket_type: contracts.SocketType = contracts.SocketType.MaxwellTemporalShape
	label: str
	
	def init(self, bl_socket: MaxwellTemporalShapeBLSocket) -> None:
		pass

####################
# - Blender Registration
####################
BL_REGISTER = [
	MaxwellTemporalShapeBLSocket,
]
