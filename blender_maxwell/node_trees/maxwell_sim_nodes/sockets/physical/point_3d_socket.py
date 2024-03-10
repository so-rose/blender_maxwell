import typing as typ

import bpy
import sympy as sp
import sympy.physics.units as spu
import pydantic as pyd

from .....utils.pydantic_sympy import SympyExpr
from .. import base
from ... import contracts as ct

class PhysicalPoint3DBLSocket(base.MaxwellSimSocket):
	socket_type = ct.SocketType.PhysicalPoint3D
	bl_label = "Volume"
	use_units = True
	
	####################
	# - Properties
	####################
	raw_value: bpy.props.FloatVectorProperty(
		name="Unitless 3D Point (global coordinate system)",
		description="Represents the unitless part of the 3D point",
		size=3,
		default=(0.0, 0.0, 0.0),
		precision=4,
		update=(lambda self, context: self.sync_prop("raw_value", context)),
	)
	
	####################
	# - Socket UI
	####################
	def draw_value(self, col: bpy.types.UILayout) -> None:
		col.prop(self, "raw_value", text="")
	
	####################
	# - Default Value
	####################
	@property
	def value(self) -> sp.MatrixBase:
		return sp.Matrix(tuple(self.raw_value)) * self.unit
	
	@value.setter
	def value(self, value: SympyExpr) -> None:
		self.raw_value = tuple(spu.convert_to(value, self.unit) / self.unit)

####################
# - Socket Configuration
####################
class PhysicalPoint3DSocketDef(pyd.BaseModel):
	socket_type: ct.SocketType = ct.SocketType.PhysicalPoint3D
	
	default_unit: typ.Any | None = None
	
	def init(self, bl_socket: PhysicalPoint3DBLSocket) -> None:
		if self.default_unit:
			bl_socket.unit = self.default_unit

####################
# - Blender Registration
####################
BL_REGISTER = [
	PhysicalPoint3DBLSocket,
]
