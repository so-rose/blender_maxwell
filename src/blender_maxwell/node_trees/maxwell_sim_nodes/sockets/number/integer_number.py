import bpy

from ... import contracts as ct
from .. import base


####################
# - Blender Socket
####################
class IntegerNumberBLSocket(base.MaxwellSimSocket):
	socket_type = ct.SocketType.IntegerNumber
	bl_label = 'Integer Number'

	####################
	# - Properties
	####################
	raw_value: bpy.props.IntProperty(
		name='Integer',
		description='Represents an integer',
		default=0,
		update=(lambda self, context: self.sync_prop('raw_value', context)),
	)

	####################
	# - Socket UI
	####################
	def draw_value(self, col: bpy.types.UILayout) -> None:
		col_row = col.row()
		col_row.prop(self, 'raw_value', text='')

	####################
	# - Default Value
	####################
	@property
	def value(self) -> int:
		return self.raw_value

	@value.setter
	def value(self, value: int) -> None:
		self.raw_value = value


####################
# - Socket Configuration
####################
class IntegerNumberSocketDef(base.SocketDef):
	socket_type: ct.SocketType = ct.SocketType.IntegerNumber

	default_value: int = 0

	def init(self, bl_socket: IntegerNumberBLSocket) -> None:
		bl_socket.value = self.default_value


####################
# - Blender Registration
####################
BL_REGISTER = [IntegerNumberBLSocket]
