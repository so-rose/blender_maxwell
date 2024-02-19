import typing as typ

import bpy
import pydantic as pyd

from .. import base
from ... import contracts

####################
# - Blender Socket
####################
class PhysicalAccelBLSocket(base.BLSocket):
	socket_type = contracts.SocketType.PhysicalAccel
	bl_label = "PhysicalAccel"
	use_units = True
	
	####################
	# - Properties
	####################
	raw_value: bpy.props.FloatProperty(
		name="Unitless Acceleration",
		description="Represents the unitless part of the acceleration",
		default=0.0,
		precision=6,
	)
	
	####################
	# - Socket UI
	####################
	def draw_label_row(self, label_col_row: bpy.types.UILayout, text: str) -> None:
		label_col_row.label(text=text)
		label_col_row.prop(self, "raw_unit", text="")
	
	def draw_value(self, col: bpy.types.UILayout) -> None:
		col_row = col.row(align=True)
		col_row.prop(self, "raw_value", text="")
	
	####################
	# - Default Value
	####################
	@property
	def default_value(self) -> None:
		return self.raw_value * self.unit
	
	@default_value.setter
	def default_value(self, value: typ.Any) -> None:
		self.raw_value = self.value_as_unit(value)

####################
# - Socket Configuration
####################
class PhysicalAccelSocketDef(pyd.BaseModel):
	socket_type: contracts.SocketType = contracts.SocketType.PhysicalAccel
	label: str
	
	def init(self, bl_socket: PhysicalAccelBLSocket) -> None:
		pass

####################
# - Blender Registration
####################
BL_REGISTER = [
	PhysicalAccelBLSocket,
]
