import typing as typ

import pydantic as pyd

from .. import base
from ... import contracts

####################
# - Blender Socket
####################
class BlenderTextBLSocket(base.BLSocket):
	socket_type = contracts.SocketType.BlenderText
	bl_label = "BlenderText"
	
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
class BlenderTextSocketDef(pyd.BaseModel):
	socket_type: contracts.SocketType = contracts.SocketType.BlenderText
	label: str
	
	def init(self, bl_socket: BlenderTextBLSocket) -> None:
		pass

####################
# - Blender Registration
####################
BL_REGISTER = [
	BlenderTextBLSocket
]
