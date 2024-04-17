import bpy

from ... import contracts as ct
from .. import base


####################
# - Blender Socket
####################
class StringBLSocket(base.MaxwellSimSocket):
	socket_type = ct.SocketType.String
	bl_label = 'String'

	####################
	# - Properties
	####################
	raw_value: bpy.props.StringProperty(
		name='String',
		description='Represents a string',
		default='',
		update=(lambda self, context: self.sync_prop('raw_value', context)),
	)

	####################
	# - Socket UI
	####################
	def draw_label_row(self, label_col_row: bpy.types.UILayout, text: str) -> None:
		label_col_row.prop(self, 'raw_value', text=text)

	####################
	# - Computation of Default Value
	####################
	@property
	def value(self) -> str:
		return self.raw_value

	@value.setter
	def value(self, value: str) -> None:
		self.raw_value = value


####################
# - Socket Configuration
####################
class StringSocketDef(base.SocketDef):
	socket_type: ct.SocketType = ct.SocketType.String

	default_text: str = ''

	def init(self, bl_socket: StringBLSocket) -> None:
		bl_socket.value = self.default_text


####################
# - Blender Registration
####################
BL_REGISTER = [
	StringBLSocket,
]



