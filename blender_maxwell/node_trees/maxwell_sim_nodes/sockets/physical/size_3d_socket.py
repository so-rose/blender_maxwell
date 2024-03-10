import typing as typ

import bpy
import sympy as sp
import sympy.physics.units as spu
import pydantic as pyd

from .....utils.pydantic_sympy import SympyExpr
from .. import base
from ... import contracts as ct

class PhysicalSize3DBLSocket(base.MaxwellSimSocket):
	socket_type = ct.SocketType.PhysicalSize3D
	bl_label = "3D Size"
	use_units = True
	
	####################
	# - Properties
	####################
	raw_value: bpy.props.FloatVectorProperty(
		name="Unitless 3D Size",
		description="Represents the unitless part of the 3D size",
		size=3,
		default=(1.0, 1.0, 1.0),
		precision=4,
		update=(lambda self, context: self.sync_prop("raw_value", context)),
	)
	
	####################
	# - Socket UI
	####################
	def draw_value(self, col: bpy.types.UILayout) -> None:
		col.prop(self, "raw_value", text="")
	
	####################
	# - Computation of Default Value
	####################
	@property
	def value(self) -> SympyExpr:
		return sp.Matrix(tuple(self.raw_value)) * self.unit
	
	@value.setter
	def value(self, value: SympyExpr) -> None:
		self.raw_value = tuple(spu.convert_to(value, self.unit) / self.unit)

####################
# - Socket Configuration
####################
class PhysicalSize3DSocketDef(pyd.BaseModel):
	socket_type: ct.SocketType = ct.SocketType.PhysicalSize3D
	
	default_unit: SympyExpr | None = None
	
	def init(self, bl_socket: PhysicalSize3DBLSocket) -> None:
		if self.default_unit:
			bl_socket.unit = self.default_unit

####################
# - Blender Registration
####################
BL_REGISTER = [
	PhysicalSize3DBLSocket,
]
