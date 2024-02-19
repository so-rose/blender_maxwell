import typing as typ

import pydantic as pyd

from .. import base
from ... import contracts

####################
# - Blender Socket
####################
class PhysicalFreqBLSocket(base.BLSocket):
	socket_type = contracts.SocketType.PhysicalFreq
	bl_label = "PhysicalFreq"
	
	####################
	# - Default Value
	####################
	@property
	def default_value(self) -> None:
		pass
	
	@default_value.setter
	def default_value(self, value: typ.Any) -> None:
		pass

####################
# - Socket Configuration
####################
class PhysicalFreqSocketDef(pyd.BaseModel):
	socket_type: contracts.SocketType = contracts.SocketType.PhysicalFreq
	label: str
	
	def init(self, bl_socket: PhysicalFreqBLSocket) -> None:
		pass

####################
# - Blender Registration
####################
BL_REGISTER = [
	PhysicalFreqBLSocket,
]
