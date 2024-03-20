import typing as typ

import bpy
import sympy as sp
import pydantic as pyd

from .....utils.pydantic_sympy import SympyExpr
from .. import base
from ... import contracts as ct


####################
# - Blender Socket
####################
class ComplexNumberBLSocket(base.MaxwellSimSocket):
	socket_type = ct.SocketType.ComplexNumber
	bl_label = 'Complex Number'

	####################
	# - Properties
	####################
	raw_value: bpy.props.FloatVectorProperty(
		name='Complex Number',
		description='Represents a complex number (real, imaginary)',
		size=2,
		default=(0.0, 0.0),
		subtype='NONE',
		update=(lambda self, context: self.sync_prop('raw_value', context)),
	)
	coord_sys: bpy.props.EnumProperty(
		name='Coordinate System',
		description='Choose between cartesian and polar form',
		items=[
			(
				'CARTESIAN',
				'Cartesian',
				'Use Cartesian Coordinates',
				'EMPTY_AXIS',
				0,
			),
			(
				'POLAR',
				'Polar',
				'Use Polar Coordinates',
				'DRIVER_ROTATIONAL_DIFFERENCE',
				1,
			),
		],
		default='CARTESIAN',
		update=lambda self, context: self._sync_coord_sys(context),
	)

	####################
	# - Socket UI
	####################
	def draw_value(self, col: bpy.types.UILayout) -> None:
		"""Draw the value of the complex number, including a toggle for
		specifying the active coordinate system.
		"""
		col_row = col.row()
		col_row.prop(self, 'raw_value', text='')
		col.prop(self, 'coord_sys', text='')

	####################
	# - Computation of Default Value
	####################
	@property
	def value(self) -> SympyExpr:
		"""Return the complex number as a sympy expression, of a form
		determined by the coordinate system.

		- Cartesian: a,b -> a + ib
		- Polar: r,t -> re^(it)

		Returns:
			The sympy expression representing the complex number.
		"""

		v1, v2 = self.raw_value

		return {
			'CARTESIAN': v1 + sp.I * v2,
			'POLAR': v1 * sp.exp(sp.I * v2),
		}[self.coord_sys]

	@value.setter
	def value(self, value: SympyExpr) -> None:
		"""Set the complex number from a sympy expression, using an internal
		representation determined by the coordinate system.

		- Cartesian: a,b -> a + ib
		- Polar: r,t -> re^(it)
		"""

		self.raw_value = {
			'CARTESIAN': (sp.re(value), sp.im(value)),
			'POLAR': (sp.Abs(value), sp.arg(value)),
		}[self.coord_sys]

	####################
	# - Internal Update Methods
	####################
	def _sync_coord_sys(self, context: bpy.types.Context):
		if self.coord_sys == 'CARTESIAN':
			r, theta_rad = self.raw_value
			self.raw_value = (
				r * sp.cos(theta_rad),
				r * sp.sin(theta_rad),
			)
		elif self.coord_sys == 'POLAR':
			x, y = self.raw_value
			cart_value = x + sp.I * y
			self.raw_value = (
				sp.Abs(cart_value),
				sp.arg(cart_value) if y != 0 else 0,
			)

		self.sync_prop('coord_sys', context)


####################
# - Socket Configuration
####################
class ComplexNumberSocketDef(pyd.BaseModel):
	socket_type: ct.SocketType = ct.SocketType.ComplexNumber

	default_value: SympyExpr = sp.S(0 + 0j)
	coord_sys: typ.Literal['CARTESIAN', 'POLAR'] = 'CARTESIAN'

	def init(self, bl_socket: ComplexNumberBLSocket) -> None:
		bl_socket.value = self.default_value
		bl_socket.coord_sys = self.coord_sys


####################
# - Blender Registration
####################
BL_REGISTER = [
	ComplexNumberBLSocket,
]
