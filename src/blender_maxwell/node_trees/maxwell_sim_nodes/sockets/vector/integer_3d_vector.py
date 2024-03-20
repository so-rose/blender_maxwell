import typing as typ

import bpy
import sympy as sp
import pydantic as pyd

from .....utils.pydantic_sympy import ConstrSympyExpr
from .. import base
from ... import contracts as ct

Integer3DVector = ConstrSympyExpr(
	allow_variables=False,
	allow_units=False,
	allowed_sets={'integer'},
	allowed_structures={'matrix'},
	allowed_matrix_shapes={(3, 1)},
)


####################
# - Blender Socket
####################
class Integer3DVectorBLSocket(base.MaxwellSimSocket):
	socket_type = ct.SocketType.Integer3DVector
	bl_label = 'Integer 3D Vector'

	####################
	# - Properties
	####################
	raw_value: bpy.props.IntVectorProperty(
		name='Int 3D Vector',
		description='Represents an integer 3D (coordinate) vector',
		size=3,
		default=(0, 0, 0),
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
	def value(self) -> Integer3DVector:
		return sp.Matrix(tuple(self.raw_value))

	@value.setter
	def value(self, value: Integer3DVector) -> None:
		self.raw_value = tuple(int(el) for el in value)


####################
# - Socket Configuration
####################
class Integer3DVectorSocketDef(pyd.BaseModel):
	socket_type: ct.SocketType = ct.SocketType.Integer3DVector

	default_value: Integer3DVector = sp.Matrix([0, 0, 0])

	def init(self, bl_socket: Integer3DVectorBLSocket) -> None:
		bl_socket.value = self.default_value


####################
# - Blender Registration
####################
BL_REGISTER = [
	Integer3DVectorBLSocket,
]
