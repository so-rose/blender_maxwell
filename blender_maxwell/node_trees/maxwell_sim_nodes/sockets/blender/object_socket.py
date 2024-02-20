import typing as typ

import bpy
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
	# - Properties
	####################
	raw_value: bpy.props.PointerProperty(
		name="Blender Object",
		description="Represents a Blender object",
		type=bpy.types.Object,
		update=(lambda self, context: self.trigger_updates()),
	)
	
	####################
	# - Default Value
	####################
	@property
	def default_value(self) -> bpy.types.Object | None:
		return self.raw_value
	
	@default_value.setter
	def default_value(self, value: bpy.types.Object) -> None:
		self.raw_value = value

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
