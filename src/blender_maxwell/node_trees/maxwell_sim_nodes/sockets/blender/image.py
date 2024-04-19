import bpy

from ... import contracts as ct
from .. import base


####################
# - Blender Socket
####################
class BlenderImageBLSocket(base.MaxwellSimSocket):
	socket_type = ct.SocketType.BlenderImage
	bl_label = 'Blender Image'

	####################
	# - Properties
	####################
	raw_value: bpy.props.PointerProperty(
		name='Blender Image',
		description='Represents a Blender Image',
		type=bpy.types.Image,
		update=(lambda self, context: self.on_prop_changed('raw_value', context)),
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
	def value(self) -> bpy.types.Image | None:
		return self.raw_value

	@value.setter
	def value(self, value: bpy.types.Image) -> None:
		self.raw_value = value


####################
# - Socket Configuration
####################
class BlenderImageSocketDef(base.SocketDef):
	socket_type: ct.SocketType = ct.SocketType.BlenderImage

	def init(self, bl_socket: BlenderImageBLSocket) -> None:
		pass


####################
# - Blender Registration
####################
BL_REGISTER = [BlenderImageBLSocket]
