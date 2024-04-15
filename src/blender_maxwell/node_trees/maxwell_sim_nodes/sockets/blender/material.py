import bpy
import pydantic as pyd

from ... import contracts as ct
from .. import base


class BlenderMaterialBLSocket(base.MaxwellSimSocket):
	socket_type = ct.SocketType.BlenderMaterial
	bl_label = 'Blender Material'

	####################
	# - Properties
	####################
	raw_value: bpy.props.PointerProperty(
		name='Blender Material',
		description='Represents a Blender material',
		type=bpy.types.Material,
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
	def value(self) -> bpy.types.Material | None:
		return self.raw_value

	@value.setter
	def value(self, value: bpy.types.Material) -> None:
		self.raw_value = value


####################
# - Socket Configuration
####################
class BlenderMaterialSocketDef(base.SocketDef):
	socket_type: ct.SocketType = ct.SocketType.BlenderMaterial

	def init(self, bl_socket: BlenderMaterialBLSocket) -> None:
		pass


####################
# - Blender Registration
####################
BL_REGISTER = [
	BlenderMaterialBLSocket,
]
