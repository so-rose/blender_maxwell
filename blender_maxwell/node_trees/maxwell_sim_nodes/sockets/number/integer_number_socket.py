import typing as typ

import bpy
import pydantic as pyd

from .. import base
from ... import contracts

####################
# - Blender Socket
####################
class IntegerNumberBLSocket(base.BLSocket):
	socket_type = contracts.SocketType.IntegerNumber
	bl_label = "IntegerNumber"
	
	####################
	# - Properties
	####################
	raw_value: bpy.props.IntProperty(
		name="Integer",
		description="Represents an integer",
		default=0,
		update=(lambda self, context: self.trigger_updates()),
	)
	
	####################
	# - Default Value
	####################
	@property
	def default_value(self) -> None:
		return self.raw_value
	
	@default_value.setter
	def default_value(self, value: typ.Any) -> None:
		self.raw_value = int(value)

####################
# - Socket Configuration
####################
class IntegerNumberSocketDef(pyd.BaseModel):
	socket_type: contracts.SocketType = contracts.SocketType.IntegerNumber
	label: str
	
	default_value: int = 0
	
	def init(self, bl_socket: IntegerNumberBLSocket) -> None:
		bl_socket.raw_value = self.default_value

####################
# - Blender Registration
####################
BL_REGISTER = [
	IntegerNumberBLSocket
]
