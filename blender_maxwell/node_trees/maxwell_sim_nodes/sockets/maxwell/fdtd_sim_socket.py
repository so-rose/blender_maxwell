import typing as typ

import bpy
import pydantic as pyd
import tidy3d as td

from .. import base
from ... import contracts

class MaxwellFDTDSimBLSocket(base.BLSocket):
	socket_type = contracts.SocketType.MaxwellFDTDSim
	bl_label = "Maxwell Source"
	
	compatible_types = {
		td.Simulation: {},
	}
	
	####################
	# - Computation of Default Value
	####################
	@property
	def default_value(self) -> None:
		return None
	
	@default_value.setter
	def default_value(self, value: typ.Any) -> None:
		pass

####################
# - Socket Configuration
####################
class MaxwellFDTDSimSocketDef(pyd.BaseModel):
	socket_type: contracts.SocketType = contracts.SocketType.MaxwellFDTDSim
	label: str
	
	def init(self, bl_socket: MaxwellFDTDSimBLSocket) -> None:
		pass

####################
# - Blender Registration
####################
BL_REGISTER = [
	MaxwellFDTDSimBLSocket,
]
