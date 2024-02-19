import typing as typ

import pydantic as pyd

from .. import base
from ... import contracts

####################
# - Blender Socket
####################
class PhysicalSpecRelPermDistBLSocket(base.BLSocket):
	socket_type = contracts.SocketType.PhysicalSpecRelPermDist
	bl_label = "PhysicalSpecRelPermDist"
	
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
class PhysicalSpecRelPermDistSocketDef(pyd.BaseModel):
	socket_type: contracts.SocketType = contracts.SocketType.PhysicalSpecRelPermDist
	label: str
	
	def init(self, bl_socket: PhysicalSpecRelPermDistBLSocket) -> None:
		pass

####################
# - Blender Registration
####################
BL_REGISTER = [
	PhysicalSpecRelPermDistBLSocket,
]
