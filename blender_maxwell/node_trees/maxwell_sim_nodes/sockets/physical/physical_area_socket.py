import typing as typ

import bpy
import sympy as sp
import sympy.physics.units as spu
import pydantic as pyd

sp.printing.str.StrPrinter._default_settings['abbrev'] = True
## When we str() a unit expression, use abbrevied units.

from .. import base
from ... import contracts

UNITS = {
	"PM_SQ": spu.pm**2,
	"A_SQ": spu.angstrom**2,
	"NM_SQ": spu.nm**2,
	"UM_SQ": spu.um**2,
	"MM_SQ": spu.mm**2,
	"CM_SQ": spu.cm**2,
	"M_SQ": spu.m**2,
}
DEFAULT_UNIT = "UM_SQ"

class PhysicalAreaBLSocket(base.BLSocket):
	socket_type = contracts.SocketType.PhysicalArea
	socket_color = (0.8, 0.5, 0.5, 1.0)
	
	bl_label = "Physical Area"
	
	compatible_types = {
		sp.Expr: {
			lambda v: v.is_real,
			lambda v: len(v.free_symbols) == 0,
			lambda v: any(
				contracts.is_exactly_expressed_as_unit(v, unit)
				for unit in UNITS.values()
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
	unit: bpy.props.EnumProperty(
		name="Unit",
		description="Choose between area units",
		items=[
			(unit_name, str(unit_value), str(unit_value))
			for unit_name, unit_value in UNITS.items()
		],
		default=DEFAULT_UNIT,
		update=lambda self, context: self._update_unit(),
	)
	unit_previous: bpy.props.StringProperty(default=DEFAULT_UNIT)
	
	####################
	# - Socket UI
	####################
	def draw_label_row(self, label_col_row: bpy.types.UILayout, text: str) -> None:
		"""Draw the value of the area, including a toggle for
		specifying the active unit.
		"""
		label_col_row.label(text=text)
		#label_col_row.prop(self, "raw_value", text="")
		label_col_row.prop(self, "unit", text="")
	
	def draw_value(self, col: bpy.types.UILayout) -> None:
		col_row = col.row(align=True)
		col_row.prop(self, "raw_value", text="")
		#col_row.prop(self, "unit", text="")
	
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
		
		return self.raw_value * UNITS[self.unit]
	
	@default_value.setter
	def default_value(self, value: typ.Any) -> None:
		"""Set the area from a sympy expression, including any required
		unit conversions to normalize the input value to the selected
		units.
		"""
		
		# (Guard) Value Compatibility
		if not self.is_compatible(value):
			msg = f"Tried setting socket ({self}) to incompatible value ({value}) of type {type(value)}"
			raise ValueError(msg)
		
		self.raw_value = spu.convert_to(
			value, UNITS[self.unit]
		) / UNITS[self.unit]
	
	####################
	# - Internal Update Methods
	####################
	def _update_unit(self):
		old_unit = UNITS[self.unit_previous] 
		new_unit = UNITS[self.unit]
		
		self.raw_value = spu.convert_to(
			self.raw_value * old_unit,
			new_unit,
		) / new_unit
		self.unit_previous = self.unit

####################
# - Socket Configuration
####################
class PhysicalAreaSocketDef(pyd.BaseModel):
	socket_type: contracts.SocketType = contracts.SocketType.PhysicalArea
	label: str
	
	default_unit: typ.Literal[
		"PM_SQ",
		"A_SQ",
		"NM_SQ",
		"UM_SQ",
		"MM_SQ",
		"CM_SQ",
		"M_SQ",
	]
	
	def init(self, bl_socket: PhysicalAreaBLSocket) -> None:
		bl_socket.unit = self.default_unit

####################
# - Blender Registration
####################
BL_REGISTER = [
	PhysicalAreaBLSocket,
]
