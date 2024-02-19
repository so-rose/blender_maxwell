import typing as typ

import pydantic as pyd

from .. import base
from ... import contracts

####################
# - Blender Socket
####################
class Real2DVectorBLSocket(base.BLSocket):
	socket_type = contracts.SocketType.Real2DVector
	bl_label = "Real2DVector"
	
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
class Real2DVectorSocketDef(pyd.BaseModel):
	socket_type: contracts.SocketType = contracts.SocketType.Real2DVector
	label: str
	
	def init(self, bl_socket: Real2DVectorBLSocket) -> None:
		pass

####################
# - Blender Registration
####################
BL_REGISTER = [
	Real2DVectorBLSocket,
]
