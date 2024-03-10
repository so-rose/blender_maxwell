import typing as typ

import bpy
import sympy as sp
import pydantic as pyd

from .....utils.pydantic_sympy import SympyExpr
from .. import base
from ... import contracts as ct

####################
# - Blender Socket
####################
class RealNumberBLSocket(base.MaxwellSimSocket):
	socket_type = ct.SocketType.RealNumber
	bl_label = "Real Number"
	
	####################
	# - Properties
	####################
	raw_value: bpy.props.FloatProperty(
		name="Real Number",
		description="Represents a real number",
		default=0.0,
		precision=6,
		update=(lambda self, context: self.sync_prop("raw_value", context)),
	)
	
	####################
	# - Socket UI
	####################
	def draw_value(self, col: bpy.types.UILayout) -> None:
		col_row = col.row()
		col_row.prop(self, "raw_value", text="")
	
	####################
	# - Computation of Default Value
	####################
	@property
	def value(self) -> float:
		return self.raw_value
	
	@value.setter
	def value(self, value: float | SympyExpr) -> None:
		if isinstance(value, float):
			self.raw_value = value
		else:
			float(value.n())

####################
# - Socket Configuration
####################
class RealNumberSocketDef(pyd.BaseModel):
	socket_type: ct.SocketType = ct.SocketType.RealNumber
	
	default_value: float = 0.0
	
	def init(self, bl_socket: RealNumberBLSocket) -> None:
		bl_socket.value = self.default_value

####################
# - Blender Registration
####################
BL_REGISTER = [
	RealNumberBLSocket,
]
