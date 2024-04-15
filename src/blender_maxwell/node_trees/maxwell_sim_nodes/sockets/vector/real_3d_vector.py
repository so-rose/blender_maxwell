import bpy
import pydantic as pyd
import sympy as sp

from .....utils.pydantic_sympy import ConstrSympyExpr
from ... import contracts as ct
from .. import base

Real3DVector = ConstrSympyExpr(
	allow_variables=False,
	allow_units=False,
	allowed_sets={'integer', 'rational', 'real'},
	allowed_structures={'matrix'},
	allowed_matrix_shapes={(3, 1)},
)


####################
# - Blender Socket
####################
class Real3DVectorBLSocket(base.MaxwellSimSocket):
	socket_type = ct.SocketType.Real3DVector
	bl_label = 'Real 3D Vector'

	####################
	# - Properties
	####################
	raw_value: bpy.props.FloatVectorProperty(
		name='Real 3D Vector',
		description='Represents a real 3D (coordinate) vector',
		size=3,
		default=(0.0, 0.0, 0.0),
		precision=4,
		update=(lambda self, context: self.sync_prop('raw_value', context)),
	)

	####################
	# - Socket UI
	####################
	def draw_value(self, col: bpy.types.UILayout) -> None:
		col.prop(self, 'raw_value', text='')

	####################
	# - Computation of Default Value
	####################
	@property
	def value(self) -> Real3DVector:
		return sp.Matrix(tuple(self.raw_value))

	@value.setter
	def value(self, value: Real3DVector) -> None:
		self.raw_value = tuple(value)


####################
# - Socket Configuration
####################
class Real3DVectorSocketDef(base.SocketDef):
	socket_type: ct.SocketType = ct.SocketType.Real3DVector

	default_value: Real3DVector = sp.Matrix([0.0, 0.0, 0.0])

	def init(self, bl_socket: Real3DVectorBLSocket) -> None:
		bl_socket.value = self.default_value


####################
# - Blender Registration
####################
BL_REGISTER = [
	Real3DVectorBLSocket,
]
