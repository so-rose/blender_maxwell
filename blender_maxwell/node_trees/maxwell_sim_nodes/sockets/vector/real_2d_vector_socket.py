import typing as typ

import bpy
import sympy as sp
import pydantic as pyd

from .....utils.pydantic_sympy import ConstrSympyExpr
from .. import base
from ... import contracts as ct

Real2DVector = ConstrSympyExpr(
	allow_variables=False,
	allow_units=False,
	allowed_sets={"integer", "rational", "real"},
	allowed_structures={"matrix"},
	allowed_matrix_shapes={(2, 1)},
)
####################
# - Blender Socket
####################
class Real2DVectorBLSocket(base.MaxwellSimSocket):
	socket_type = ct.SocketType.Real2DVector
	bl_label = "Real2DVector"
	
	####################
	# - Properties
	####################
	raw_value: bpy.props.FloatVectorProperty(
		name="Unitless 2D Vector (global coordinate system)",
		description="Represents a real 2D (coordinate) vector",
		size=2,
		default=(0.0, 0.0),
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
	def value(self) -> Real2DVector:
		return sp.Matrix(tuple(self.raw_value))
	
	@value.setter
	def value(self, value: Real2DVector) -> None:
		self.raw_value = tuple(value)

####################
# - Socket Configuration
####################
class Real2DVectorSocketDef(pyd.BaseModel):
	socket_type: ct.SocketType = ct.SocketType.Real2DVector
	
	default_value: Real2DVector = sp.Matrix([0.0, 0.0])
	
	def init(self, bl_socket: Real2DVectorBLSocket) -> None:
		bl_socket.value = self.default_value

####################
# - Blender Registration
####################
BL_REGISTER = [
	Real2DVectorBLSocket,
]
