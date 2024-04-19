import bpy
import sympy as sp

import blender_maxwell.utils.extra_sympy_units as spux

from ... import contracts as ct
from .. import base


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
		update=(lambda self, context: self.on_prop_changed('raw_value', context)),
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
	def value(self) -> spux.Real3DVector:
		return sp.Matrix(tuple(self.raw_value))

	@value.setter
	def value(self, value: spux.Real3DVector) -> None:
		self.raw_value = tuple(value)


####################
# - Socket Configuration
####################
class Real3DVectorSocketDef(base.SocketDef):
	socket_type: ct.SocketType = ct.SocketType.Real3DVector

	default_value: spux.Real3DVector = sp.Matrix([0.0, 0.0, 0.0])

	def init(self, bl_socket: Real3DVectorBLSocket) -> None:
		bl_socket.value = self.default_value


####################
# - Blender Registration
####################
BL_REGISTER = [
	Real3DVectorBLSocket,
]
