import typing as typ

import pydantic as pyd

from .. import base
from ... import contracts

####################
# - Blender Socket
####################
class BlenderCollectionBLSocket(base.BLSocket):
	socket_type = contracts.SocketType.BlenderCollection
	bl_label = "BlenderCollection"
	
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
class BlenderCollectionSocketDef(pyd.BaseModel):
	socket_type: contracts.SocketType = contracts.SocketType.BlenderCollection
	label: str
	
	def init(self, bl_socket: BlenderCollectionBLSocket) -> None:
		pass

####################
# - Blender Registration
####################
BL_REGISTER = [
	BlenderCollectionBLSocket
]
