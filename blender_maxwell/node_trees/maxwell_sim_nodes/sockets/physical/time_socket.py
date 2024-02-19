import typing as typ

import pydantic as pyd

from .. import base
from ... import contracts

####################
# - Blender Socket
####################
class PhysicalTimeBLSocket(base.BLSocket):
	socket_type = contracts.SocketType.PhysicalTime
	bl_label = "PhysicalTime"
	
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
class PhysicalTimeSocketDef(pyd.BaseModel):
	socket_type: contracts.SocketType = contracts.SocketType.PhysicalTime
	label: str
	
	def init(self, bl_socket: PhysicalTimeBLSocket) -> None:
		pass

####################
# - Blender Registration
####################
BL_REGISTER = [
	PhysicalTimeBLSocket,
]
