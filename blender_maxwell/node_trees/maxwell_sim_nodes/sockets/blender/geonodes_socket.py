import typing as typ

import pydantic as pyd

from .. import base
from ... import contracts

####################
# - Blender Socket
####################
class BlenderGeoNodesBLSocket(base.BLSocket):
	socket_type = contracts.SocketType.BlenderGeoNodes
	bl_label = "BlenderGeoNodes"
	
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
class BlenderGeoNodesSocketDef(pyd.BaseModel):
	socket_type: contracts.SocketType = contracts.SocketType.BlenderGeoNodes
	label: str
	
	def init(self, bl_socket: BlenderGeoNodesBLSocket) -> None:
		pass

####################
# - Blender Registration
####################
BL_REGISTER = [
	BlenderGeoNodesBLSocket
]
