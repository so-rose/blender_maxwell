import typing as typ

import bpy
import sympy as sp
import pydantic as pyd

from .. import base
from ... import contracts

####################
# - Blender Socket
####################
class BoolBLSocket(base.BLSocket):
	socket_type = contracts.SocketType.Bool
	bl_label = "Bool"
	
	compatible_types = {
		bool: {},
	}
	
	####################
	# - Properties
	####################
	raw_value: bpy.props.BoolProperty(
		name="Boolean",
		description="Represents a boolean",
		default=False,
	)
	
	####################
	# - Socket UI
	####################
	def draw_label_row(self, label_col_row: bpy.types.UILayout, text: str) -> None:
		label_col_row.label(text=text)
		label_col_row.prop(self, "raw_value", text="")
	
	def draw_value(self, label_col_row: bpy.types.UILayout) -> None:
		pass
	
	####################
	# - Computation of Default Value
	####################
	@property
	def default_value(self) -> str:
		return self.raw_value
	
	@default_value.setter
	def default_value(self, value: typ.Any) -> None:
		# (Guard) Value Compatibility
		if not self.is_compatible(value):
			msg = f"Tried setting socket ({self}) to incompatible value ({value}) of type {type(value)}"
			raise ValueError(msg)
		
		self.raw_value = bool(value)

####################
# - Socket Configuration
####################
class BoolSocketDef(pyd.BaseModel):
	socket_type: contracts.SocketType = contracts.SocketType.Bool
	label: str
	
	default_value: bool = False
	
	def init(self, bl_socket: BoolBLSocket) -> None:
		bl_socket.raw_value = self.default_value

####################
# - Blender Registration
####################
BL_REGISTER = [
	BoolBLSocket,
]
