import typing as typ

import bpy
import sympy as sp
import pydantic as pyd

from .. import base
from ... import contracts

####################
# - Blender Socket
####################
class RealNumberBLSocket(base.BLSocket):
	socket_type = contracts.SocketType.RealNumber
	bl_label = "Real Number"
	
	compatible_types = {
		float: {},
		sp.Expr: {
			lambda self, v: v.is_real,
			lambda self, v: len(v.free_symbols) == 0,
		},
	}
	
	####################
	# - Properties
	####################
	raw_value: bpy.props.FloatProperty(
		name="Real Number",
		description="Represents a real number",
		default=0.0,
		precision=6,
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
	def default_value(self) -> float:
		"""Return the real number.
		
		Returns:
			The real number as a float.
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
		
		self.raw_value = float(value)

####################
# - Socket Configuration
####################
class RealNumberSocketDef(pyd.BaseModel):
	socket_type: contracts.SocketType = contracts.SocketType.RealNumber
	label: str
	
	default_value: float = 0.0
	
	def init(self, bl_socket: RealNumberBLSocket) -> None:
		bl_socket.default_value = self.default_value

####################
# - Blender Registration
####################
BL_REGISTER = [
	RealNumberBLSocket,
]
