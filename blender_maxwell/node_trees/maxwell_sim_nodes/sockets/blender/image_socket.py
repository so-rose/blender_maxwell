import typing as typ

import pydantic as pyd

from .. import base
from ... import contracts

####################
# - Blender Socket
####################
class BlenderImageBLSocket(base.BLSocket):
	socket_type = contracts.SocketType.BlenderImage
	bl_label = "BlenderImage"
	
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
class BlenderImageSocketDef(pyd.BaseModel):
	socket_type: contracts.SocketType = contracts.SocketType.BlenderImage
	label: str
	
	def init(self, bl_socket: BlenderImageBLSocket) -> None:
		pass

####################
# - Blender Registration
####################
BL_REGISTER = [
	BlenderImageBLSocket
]
