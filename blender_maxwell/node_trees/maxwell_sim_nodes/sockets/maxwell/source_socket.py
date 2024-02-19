import typing as typ

import bpy
import pydantic as pyd
import tidy3d as td

from .. import base
from ... import contracts

class MaxwellSourceBLSocket(base.BLSocket):
	socket_type = contracts.SocketType.MaxwellSource
	bl_label = "Maxwell Source"
	
	compatible_types = {
		td.components.base_sim.source.AbstractSource: {}
	}
	
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
class MaxwellSourceSocketDef(pyd.BaseModel):
	socket_type: contracts.SocketType = contracts.SocketType.MaxwellSource
	label: str
	
	def init(self, bl_socket: MaxwellSourceBLSocket) -> None:
		pass

####################
# - Blender Registration
####################
BL_REGISTER = [
	MaxwellSourceBLSocket,
]
