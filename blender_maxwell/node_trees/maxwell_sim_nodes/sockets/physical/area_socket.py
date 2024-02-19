import typing as typ

import bpy
import sympy as sp
import sympy.physics.units as spu
import pydantic as pyd

from .. import base
from ... import contracts

class PhysicalAreaBLSocket(base.BLSocket):
	socket_type = contracts.SocketType.PhysicalArea
	bl_label = "Physical Area"
	use_units = True
	
	compatible_types = {
		sp.Expr: {
			lambda self, v: v.is_real,
			lambda self, v: len(v.free_symbols) == 0,
			lambda self, v: any(
				contracts.is_exactly_expressed_as_unit(v, unit)
				for unit in self.units.values()
			)
		},
	}
	
	####################
	# - Properties
	####################
	raw_value: bpy.props.FloatProperty(
		name="Unitless Area",
		description="Represents the unitless part of the area",
		default=0.0,
		precision=6,
	)
	
	####################
	# - Socket UI
	####################
	def draw_label_row(self, label_col_row: bpy.types.UILayout, text: str) -> None:
		"""Draw the value of the area, including a toggle for
		specifying the active unit.
		"""
		label_col_row.label(text=text)
		label_col_row.prop(self, "raw_unit", text="")
	
	def draw_value(self, col: bpy.types.UILayout) -> None:
		col_row = col.row(align=True)
		col_row.prop(self, "raw_value", text="")
	
	####################
	# - Computation of Default Value
	####################
	@property
	def default_value(self) -> sp.Expr:
		"""Return the area as a sympy expression, which is a pure real
		number perfectly expressed as the active unit.
		
		Returns:
			The area as a sympy expression (with units).
		"""
		
		return self.raw_value * self.unit
	
	@default_value.setter
	def default_value(self, value: typ.Any) -> None:
		"""Set the area from a sympy expression, including any required
		unit conversions to normalize the input value to the selected
		units.
		"""
		
		self.raw_value = self.value_as_unit(value)

####################
# - Socket Configuration
####################
class PhysicalAreaSocketDef(pyd.BaseModel):
	socket_type: contracts.SocketType = contracts.SocketType.PhysicalArea
	label: str
	
	default_unit: typ.Any | None = None
	
	def init(self, bl_socket: PhysicalAreaBLSocket) -> None:
		if self.default_unit:
			bl_socket.unit = self.default_unit

####################
# - Blender Registration
####################
BL_REGISTER = [
	PhysicalAreaBLSocket,
]
