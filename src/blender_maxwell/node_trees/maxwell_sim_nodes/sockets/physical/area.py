import typing as typ

import bpy
import pydantic as pyd
import sympy as sp

from ... import contracts as ct
from .. import base


class PhysicalAreaBLSocket(base.MaxwellSimSocket):
	socket_type = ct.SocketType.PhysicalArea
	bl_label = 'Physical Area'
	use_units = True

	####################
	# - Properties
	####################
	raw_value: bpy.props.FloatProperty(
		name='Unitless Area',
		description='Represents the unitless part of the area',
		default=0.0,
		precision=6,
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
	def default_value(self) -> sp.Expr:
		"""Return the area as a sympy expression, which is a pure real
		number perfectly expressed as the active unit.

		Returns:
			The area as a sympy expression (with units).
		"""
		return self.raw_value * self.unit

	@default_value.setter
	def default_value(self, value: typ.Any) -> None:
		"""Set the area from a sympy expression, including any required
		unit conversions to normalize the input value to the selected
		units.
		"""
		self.raw_value = self.value_as_unit(value)


####################
# - Socket Configuration
####################
class PhysicalAreaSocketDef(base.SocketDef):
	socket_type: ct.SocketType = ct.SocketType.PhysicalArea

	default_unit: typ.Any | None = None

	def init(self, bl_socket: PhysicalAreaBLSocket) -> None:
		if self.default_unit:
			bl_socket.unit = self.default_unit


####################
# - Blender Registration
####################
BL_REGISTER = [
	PhysicalAreaBLSocket,
]
