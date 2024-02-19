import typing as typ

import pydantic as pyd

from .. import base
from ... import contracts

####################
# - Blender Socket
####################
class PhysicalSpeedBLSocket(base.BLSocket):
	socket_type = contracts.SocketType.PhysicalSpeed
	bl_label = "PhysicalSpeed"
	
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
class PhysicalSpeedSocketDef(pyd.BaseModel):
	socket_type: contracts.SocketType = contracts.SocketType.PhysicalSpeed
	label: str
	
	def init(self, bl_socket: PhysicalSpeedBLSocket) -> None:
		pass

####################
# - Blender Registration
####################
BL_REGISTER = [
	PhysicalSpeedBLSocket,
]
