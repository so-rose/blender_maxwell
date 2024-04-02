import bpy
import pydantic as pyd

from ... import contracts as ct
from .. import base


####################
# - Blender Socket
####################
class BlenderCollectionBLSocket(base.MaxwellSimSocket):
	socket_type = ct.SocketType.BlenderCollection
	bl_label = 'Blender Collection'

	####################
	# - Properties
	####################
	raw_value: bpy.props.PointerProperty(
		name='Blender Collection',
		description='A Blender collection',
		type=bpy.types.Collection,
		update=(lambda self, context: self.sync_prop('raw_value', context)),
	)

	####################
	# - UI
	####################
	def draw_value(self, col: bpy.types.UILayout) -> None:
		col.prop(self, 'raw_value', text='')

	####################
	# - Default Value
	####################
	@property
	def value(self) -> bpy.types.Collection | None:
		return self.raw_value

	@value.setter
	def value(self, value: bpy.types.Collection) -> None:
		self.raw_value = value


####################
# - Socket Configuration
####################
class BlenderCollectionSocketDef(pyd.BaseModel):
	socket_type: ct.SocketType = ct.SocketType.BlenderCollection

	def init(self, bl_socket: BlenderCollectionBLSocket) -> None:
		pass


####################
# - Blender Registration
####################
BL_REGISTER = [BlenderCollectionBLSocket]
