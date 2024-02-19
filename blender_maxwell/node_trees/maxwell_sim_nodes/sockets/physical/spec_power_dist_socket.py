import typing as typ

import pydantic as pyd

from .. import base
from ... import contracts

####################
# - Blender Socket
####################
class PhysicalSpecPowerDistBLSocket(base.BLSocket):
	socket_type = contracts.SocketType.PhysicalSpecPowerDist
	bl_label = "PhysicalSpecPowerDist"
	
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
class PhysicalSpecPowerDistSocketDef(pyd.BaseModel):
	socket_type: contracts.SocketType = contracts.SocketType.PhysicalSpecPowerDist
	label: str
	
	def init(self, bl_socket: PhysicalSpecPowerDistBLSocket) -> None:
		pass

####################
# - Blender Registration
####################
BL_REGISTER = [
	PhysicalSpecPowerDistBLSocket,
]
