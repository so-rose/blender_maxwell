import typing as typ

import pydantic as pyd

from .. import base
from ... import contracts

####################
# - Blender Socket
####################
class BlenderObjectBLSocket(base.BLSocket):
	socket_type = contracts.SocketType.BlenderObject
	bl_label = "BlenderObject"
	
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
class BlenderObjectSocketDef(pyd.BaseModel):
	socket_type: contracts.SocketType = contracts.SocketType.BlenderObject
	label: str
	
	def init(self, bl_socket: BlenderObjectBLSocket) -> None:
		pass

####################
# - Blender Registration
####################
BL_REGISTER = [
	BlenderObjectBLSocket
]
