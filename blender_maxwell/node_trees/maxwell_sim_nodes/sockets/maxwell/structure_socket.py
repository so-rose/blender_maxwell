import typing as typ

import bpy
import pydantic as pyd
import tidy3d as td

from .. import base
from ... import contracts

class MaxwellStructureBLSocket(base.BLSocket):
	socket_type = contracts.SocketType.MaxwellStructure
	bl_label = "Maxwell Structure"
	
	compatible_types = {
		td.components.structure.AbstractStructure: {}
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
class MaxwellStructureSocketDef(pyd.BaseModel):
	socket_type: contracts.SocketType = contracts.SocketType.MaxwellStructure
	label: str
	
	def init(self, bl_socket: MaxwellStructureBLSocket) -> None:
		pass

####################
# - Blender Registration
####################
BL_REGISTER = [
	MaxwellStructureBLSocket,
]
