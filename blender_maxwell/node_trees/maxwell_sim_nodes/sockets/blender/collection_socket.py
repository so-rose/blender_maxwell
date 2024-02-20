import typing as typ

import bpy
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
	# - Properties
	####################
	raw_value: bpy.props.PointerProperty(
		name="Blender Collection",
		description="Represents a Blender collection",
		type=bpy.types.Collection,
		update=(lambda self, context: self.trigger_updates()),
	)
	
	####################
	# - Default Value
	####################
	@property
	def default_value(self) -> bpy.types.Collection | None:
		return self.raw_value
	
	@default_value.setter
	def default_value(self, value: bpy.types.Collection) -> None:
		self.raw_value = value

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
