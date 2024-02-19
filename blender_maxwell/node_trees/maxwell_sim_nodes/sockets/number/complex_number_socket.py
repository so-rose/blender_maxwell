import typing as typ

import bpy
import sympy as sp
import pydantic as pyd

from .. import base
from ... import contracts

####################
# - Blender Socket
####################
class ComplexNumberBLSocket(base.BLSocket):
	socket_type = contracts.SocketType.ComplexNumber
	bl_label = "Complex Number"
	
	compatible_types = {
		complex: {},
		sp.Expr: {
			lambda self, v: v.is_complex,
			lambda self, v: len(v.free_symbols) == 0,
		},
	}
	
	####################
	# - Properties
	####################
	raw_value: bpy.props.FloatVectorProperty(
		name="Complex Number",
		description="Represents a complex number (real, imaginary)",
		size=2,
		default=(0.0, 0.0),
		subtype='NONE'
	)
	coord_sys: bpy.props.EnumProperty(
		name="Coordinate System",
		description="Choose between cartesian and polar form",
		items=[
			("CARTESIAN", "Cartesian", "Use Cartesian Coordinates", "EMPTY_AXIS", 0),
			("POLAR", "Polar", "Use Polar Coordinates", "DRIVER_ROTATIONAL_DIFFERENCE", 1),
		],
		default="CARTESIAN",
		update=lambda self, context: self._update_coord_sys(),
	)
	
	####################
	# - Socket UI
	####################
	def draw_value(self, col: bpy.types.UILayout) -> None:
		"""Draw the value of the complex number, including a toggle for
		specifying the active coordinate system.
		"""
		col_row = col.row()
		col_row.prop(self, "raw_value", text="")
		col.prop(self, "coord_sys", text="")
	
	def draw_preview(self, col_box: bpy.types.UILayout) -> None:
		"""Draw a live-preview value for the complex number, into the
		given preview box.
		
		- Cartesian: a,b -> a + ib
		- Polar: r,t -> re^(it)
		
		Returns:
			The sympy expression representing the complex number.
		"""
		if self.coord_sys == "CARTESIAN":
			text = f"= {self.default_value.n(2)}"
		
		elif self.coord_sys == "POLAR":
			r = sp.Abs(self.default_value).n(2)
			theta_rad = sp.arg(self.default_value).n(2)
			text = f"= {r*sp.exp(sp.I*theta_rad)}"
		
		else:
			raise RuntimeError("Invalid coordinate system for complex number")
			
		col_box.label(text=text)
	
	####################
	# - Computation of Default Value
	####################
	@property
	def default_value(self) -> sp.Expr:
		"""Return the complex number as a sympy expression, of a form
		determined by the coordinate system.
		
		- Cartesian: a,b -> a + ib
		- Polar: r,t -> re^(it)
		
		Returns:
			The sympy expression representing the complex number.
		"""
		
		v1, v2 = self.raw_value
		
		return {
			"CARTESIAN": v1 + sp.I*v2,
			"POLAR": v1 * sp.exp(sp.I*v2),
		}[self.coord_sys]
	
	@default_value.setter
	def default_value(self, value: typ.Any) -> None:
		"""Set the complex number from a sympy expression, using an internal
		representation determined by the coordinate system.
		
		- Cartesian: a,b -> a + ib
		- Polar: r,t -> re^(it)
		"""
		
		# (Guard) Value Compatibility
		if not self.is_compatible(value):
			msg = f"Tried setting socket ({self}) to incompatible value ({value}) of type {type(value)}"
			raise ValueError(msg)
		
		self.raw_value = {
			"CARTESIAN": (sp.re(value), sp.im(value)),
			"POLAR": (sp.Abs(value), sp.arg(value)),
		}[self.coord_sys]
	
	####################
	# - Internal Update Methods
	####################
	def _update_coord_sys(self):
		if self.coord_sys == "CARTESIAN":
			r, theta_rad = self.raw_value
			self.raw_value = (
				r * sp.cos(theta_rad),
				r * sp.sin(theta_rad),
			)
		elif self.coord_sys == "POLAR":
			x, y = self.raw_value
			cart_value = x + sp.I*y
			self.raw_value = (
				sp.Abs(cart_value),
				sp.arg(cart_value) if y != 0 else 0,
			)

####################
# - Socket Configuration
####################
class ComplexNumberSocketDef(pyd.BaseModel):
	socket_type: contracts.SocketType = contracts.SocketType.ComplexNumber
	label: str
	
	preview: bool = False
	coord_sys: typ.Literal["CARTESIAN", "POLAR"] = "CARTESIAN"
	
	def init(self, bl_socket: ComplexNumberBLSocket) -> None:
		bl_socket.preview_active = self.preview
		bl_socket.coord_sys = self.coord_sys

####################
# - Blender Registration
####################
BL_REGISTER = [
	ComplexNumberBLSocket,
]
