import typing as typ

import bpy
import sympy as sp
import pydantic as pyd

from .. import base
from ... import contracts as ct

####################
# - Blender Socket
####################
class TextBLSocket(base.MaxwellSimSocket):
	socket_type = ct.SocketType.Text
	bl_label = "Text"
	
	####################
	# - Properties
	####################
	raw_value: bpy.props.StringProperty(
		name="Text",
		description="Represents some text",
		default="",
		update=(lambda self, context: self.sync_prop("raw_value", context)),
	)
	
	####################
	# - Socket UI
	####################
	def draw_label_row(self, label_col_row: bpy.types.UILayout, text: str) -> None:
		"""Draw the value of the real number.
		"""
		label_col_row.prop(self, "raw_value", text=text)
	
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
class TextSocketDef(pyd.BaseModel):
	socket_type: ct.SocketType = ct.SocketType.Text
	
	default_text: str = ""
	
	def init(self, bl_socket: TextBLSocket) -> None:
		bl_socket.value = self.default_text

####################
# - Blender Registration
####################
BL_REGISTER = [
	TextBLSocket,
]
