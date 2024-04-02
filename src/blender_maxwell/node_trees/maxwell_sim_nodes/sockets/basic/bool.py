
import bpy
import pydantic as pyd

from ... import contracts as ct
from .. import base


####################
# - Blender Socket
####################
class BoolBLSocket(base.MaxwellSimSocket):
	socket_type = ct.SocketType.Bool
	bl_label = 'Bool'

	####################
	# - Properties
	####################
	raw_value: bpy.props.BoolProperty(
		name='Boolean',
		description='Represents a boolean value',
		default=False,
		update=(lambda self, context: self.sync_prop('raw_value', context)),
	)

	####################
	# - Socket UI
	####################
	def draw_label_row(
		self, label_col_row: bpy.types.UILayout, text: str
	) -> None:
		label_col_row.label(text=text)
		label_col_row.prop(self, 'raw_value', text='')

	####################
	# - Computation of Default Value
	####################
	@property
	def value(self) -> bool:
		return self.raw_value

	@value.setter
	def value(self, value: bool) -> None:
		self.raw_value = value


####################
# - Socket Configuration
####################
class BoolSocketDef(pyd.BaseModel):
	socket_type: ct.SocketType = ct.SocketType.Bool

	default_value: bool = False

	def init(self, bl_socket: BoolBLSocket) -> None:
		bl_socket.value = self.default_value


####################
# - Blender Registration
####################
BL_REGISTER = [
	BoolBLSocket,
]
