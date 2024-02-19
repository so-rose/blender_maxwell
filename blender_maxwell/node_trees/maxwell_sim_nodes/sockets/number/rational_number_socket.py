import typing as typ

import pydantic as pyd

from .. import base
from ... import contracts

####################
# - Blender Socket
####################
class RationalNumberBLSocket(base.BLSocket):
	socket_type = contracts.SocketType.RationalNumber
	bl_label = "Rational Number"
	
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
class RationalNumberSocketDef(pyd.BaseModel):
	socket_type: contracts.SocketType = contracts.SocketType.RationalNumber
	label: str
	
	def init(self, bl_socket: RationalNumberBLSocket) -> None:
		pass

####################
# - Blender Registration
####################
BL_REGISTER = [
	RationalNumberBLSocket,
]
