import typing as typ

import bpy
import sympy as sp
import pydantic as pyd

from .. import base
from ... import contracts

####################
# - Blender Socket
####################
class TextBLSocket(base.BLSocket):
	socket_type = contracts.SocketType.Text
	socket_color = (0.2, 0.2, 0.2, 1.0)
	
	bl_label = "Text"
	
	compatible_types = {
		str: {},
	}
	
	####################
	# - Properties
	####################
	raw_value: bpy.props.StringProperty(
		name="Text",
		description="Represents some text",
		default="",
		update=(lambda self, context: self.trigger_updates()),
	)
	
	####################
	# - Socket UI
	####################
	def draw_label_row(self, label_col_row: bpy.types.UILayout, text: str) -> None:
		"""Draw the value of the real number.
		"""
		label_col_row.prop(self, "raw_value", text=text)
	
	def draw_value(self, label_col_row: bpy.types.UILayout) -> None:
		pass
	
	####################
	# - Computation of Default Value
	####################
	@property
	def default_value(self) -> str:
		"""Return the text.
		
		Returns:
			The text as a string.
		"""
		
		return self.raw_value
	
	@default_value.setter
	def default_value(self, value: typ.Any) -> None:
		"""Set the real number from some compatible type, namely
		real sympy expressions with no symbols, or floats.
		"""
		
		# (Guard) Value Compatibility
		if not self.is_compatible(value):
			msg = f"Tried setting socket ({self}) to incompatible value ({value}) of type {type(value)}"
			raise ValueError(msg)
		
		self.raw_value = str(value)

####################
# - Socket Configuration
####################
class TextSocketDef(pyd.BaseModel):
	socket_type: contracts.SocketType = contracts.SocketType.Text
	label: str
	
	def init(self, bl_socket: TextBLSocket) -> None:
		pass

####################
# - Blender Registration
####################
BL_REGISTER = [
	TextBLSocket,
]
