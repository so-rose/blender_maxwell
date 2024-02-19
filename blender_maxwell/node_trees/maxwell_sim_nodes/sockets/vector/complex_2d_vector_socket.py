import typing as typ

import pydantic as pyd

from .. import base
from ... import contracts

####################
# - Blender Socket
####################
class Complex2DVectorBLSocket(base.BLSocket):
	socket_type = contracts.SocketType.Complex2DVector
	bl_label = "Complex2DVector"
	
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
class Complex2DVectorSocketDef(pyd.BaseModel):
	socket_type: contracts.SocketType = contracts.SocketType.Complex2DVector
	label: str
	
	def init(self, bl_socket: Complex2DVectorBLSocket) -> None:
		pass

####################
# - Blender Registration
####################
BL_REGISTER = [
	Complex2DVectorBLSocket,
]
