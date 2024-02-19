import typing as typ

import pydantic as pyd

from .. import base
from ... import contracts

####################
# - Blender Socket
####################
class IntegerNumberBLSocket(base.BLSocket):
	socket_type = contracts.SocketType.IntegerNumber
	bl_label = "IntegerNumber"
	
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
class IntegerNumberSocketDef(pyd.BaseModel):
	socket_type: contracts.SocketType = contracts.SocketType.IntegerNumber
	label: str
	
	def init(self, bl_socket: IntegerNumberBLSocket) -> None:
		pass

####################
# - Blender Registration
####################
BL_REGISTER = [
	IntegerNumberBLSocket
]
