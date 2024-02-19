import typing as typ

import pydantic as pyd

from .. import base
from ... import contracts

####################
# - Blender Socket
####################
class Real3DVectorBLSocket(base.BLSocket):
	socket_type = contracts.SocketType.Real3DVector
	bl_label = "Real3DVector"
	
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
class Real3DVectorSocketDef(pyd.BaseModel):
	socket_type: contracts.SocketType = contracts.SocketType.Real3DVector
	label: str
	
	def init(self, bl_socket: Real3DVectorBLSocket) -> None:
		pass

####################
# - Blender Registration
####################
BL_REGISTER = [
	Real3DVectorBLSocket,
]
