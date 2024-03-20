import typing as typ

import bpy
import pydantic as pyd

from .. import base
from ... import contracts as ct


####################
# - Blender Socket
####################
class BlenderTextBLSocket(base.MaxwellSimSocket):
	socket_type = ct.SocketType.BlenderText
	bl_label = 'Blender Text'

	####################
	# - Properties
	####################
	raw_value: bpy.props.PointerProperty(
		name='Blender Text',
		description='Represents a Blender text datablock',
		type=bpy.types.Text,
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
	def value(self) -> bpy.types.Text:
		return self.raw_value

	@value.setter
	def value(self, value: bpy.types.Text) -> None:
		self.raw_value = value


####################
# - Socket Configuration
####################
class BlenderTextSocketDef(pyd.BaseModel):
	socket_type: ct.SocketType = ct.SocketType.BlenderText

	def init(self, bl_socket: BlenderTextBLSocket) -> None:
		pass


####################
# - Blender Registration
####################
BL_REGISTER = [BlenderTextBLSocket]
