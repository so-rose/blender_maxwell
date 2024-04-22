import typing as typ

import bpy
import sympy as sp

from blender_maxwell.utils import extra_sympy_units as spux

from ... import contracts as ct
from .. import base


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
		update=(lambda self, context: self.on_prop_changed('raw_value', context)),
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
		update=lambda self, context: self.on_coord_sys_changed(context),
	)

	####################
	# - Event Methods
	####################
	def on_coord_sys_changed(self, context: bpy.types.Context):
		r"""Transforms values when the coordinate system changes.

		Notes:
			Cartesian coordinates with $y=0$ has no corresponding $\theta$
			Therefore, we manually set $\theta=0$.

		"""
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
				float(sp.Abs(cart_value)),
				float(sp.arg(cart_value)) if y != 0 else float(0),
			)

		self.on_prop_changed('coord_sys', context)

	####################
	# - Socket UI
	####################
	def draw_value(self, col: bpy.types.UILayout) -> None:
		"""Draw the value of the complex number, including a toggle for specifying the active coordinate system."""
		# Value Row
		row = col.row()
		row.prop(self, 'raw_value', text='')

		# Coordinate System Dropdown
		col.prop(self, 'coord_sys', text='')

	####################
	# - Computation of Default Value
	####################
	@property
	def value(self) -> spux.ComplexNumber:
		"""Return the complex number as a sympy expression, of a form determined by the coordinate system.

		- **Cartesian**: $(a,b) -> a + ib$
		- **Polar**: $(r,t) -> re^(it)$

		Returns:
			The complex number as a `sympy` type.
		"""
		v1, v2 = self.raw_value

		return {
			'CARTESIAN': v1 + sp.I * v2,
			'POLAR': v1 * sp.exp(sp.I * v2),
		}[self.coord_sys]

	@value.setter
	def value(self, value: spux.ComplexNumber) -> None:
		"""Set the complex number from a sympy expression, by numerically simplifying it into coordinate-system determined components.

		- **Cartesian**: $(a,b) -> a + ib$
		- **Polar**: $(r,t) -> re^(it)$

		Parameters:
			value: The complex number as a `sympy` type.
		"""
		self.raw_value = {
			'CARTESIAN': (float(sp.re(value)), float(sp.im(value))),
			'POLAR': (float(sp.Abs(value)), float(sp.arg(value))),
		}[self.coord_sys]


####################
# - Socket Configuration
####################
class ComplexNumberSocketDef(base.SocketDef):
	socket_type: ct.SocketType = ct.SocketType.ComplexNumber

	default_value: spux.ComplexNumber = sp.S(0)
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
