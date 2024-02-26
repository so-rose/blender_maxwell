import typing as typ

import bpy
import pydantic as pyd

from .. import base
from ... import contracts

####################
# - Blender Socket
####################
class PhysicalPolBLSocket(base.BLSocket):
	socket_type = contracts.SocketType.PhysicalPol
	bl_label = "PhysicalPol"
	
	####################
	# - Properties
	####################
	default_choice: bpy.props.EnumProperty(
		name="Bound Face",
		description="A choice of default boundary face",
		items=[
			("EX", "Ex", "Linear x-pol of E field"),
			("EY", "Ey", "Linear y-pol of E field"),
			("EZ", "Ez", "Linear z-pol of E field"),
			("HX", "Hx", "Linear x-pol of H field"),
			("HY", "Hy", "Linear x-pol of H field"),
			("HZ", "Hz", "Linear x-pol of H field"),
		],
		default="EX",
		update=(lambda self, context: self.trigger_updates()),
	)
	
	####################
	# - UI
	####################
	def draw_value(self, col: bpy.types.UILayout) -> None:
		col_row = col.row(align=True)
		col_row.prop(self, "default_choice", text="")
	
	####################
	# - Default Value
	####################
	@property
	def default_value(self) -> str:
		return {
			"EX": "Ex",
			"EY": "Ey",
			"EZ": "Ez",
			"HX": "Hx",
			"HY": "Hy",
			"HZ": "Hz",
		}[self.default_choice]
	
	@default_value.setter
	def default_value(self, value: typ.Any) -> None:
		pass

####################
# - Socket Configuration
####################
class PhysicalPolSocketDef(pyd.BaseModel):
	socket_type: contracts.SocketType = contracts.SocketType.PhysicalPol
	label: str
	
	def init(self, bl_socket: PhysicalPolBLSocket) -> None:
		pass

####################
# - Blender Registration
####################
BL_REGISTER = [
	PhysicalPolBLSocket,
]
