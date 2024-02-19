import typing as typ

import pydantic as pyd

from .. import base
from ... import contracts

####################
# - Blender Socket
####################
class Complex3DVectorBLSocket(base.BLSocket):
	socket_type = contracts.SocketType.Complex3DVector
	bl_label = "Complex3DVector"
	
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
class Complex3DVectorSocketDef(pyd.BaseModel):
	socket_type: contracts.SocketType = contracts.SocketType.Complex3DVector
	label: str
	
	def init(self, bl_socket: Complex3DVectorBLSocket) -> None:
		pass

####################
# - Blender Registration
####################
BL_REGISTER = [
	Complex3DVectorBLSocket,
]
