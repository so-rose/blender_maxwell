import typing as typ

import bpy
import pydantic as pyd
import sympy as sp
import sympy.physics.units as spu
import tidy3d as td
import scipy as sc

from .....utils.pydantic_sympy import ConstrSympyExpr, Complex
from .. import base
from ... import contracts as ct

VAC_SPEED_OF_LIGHT = sc.constants.speed_of_light * spu.meter / spu.second


class MaxwellMediumBLSocket(base.MaxwellSimSocket):
	socket_type = ct.SocketType.MaxwellMedium
	bl_label = 'Maxwell Medium'
	use_units = True

	####################
	# - Properties
	####################
	wl: bpy.props.FloatProperty(
		name='WL',
		description='WL to evaluate conductivity at',
		default=500.0,
		precision=4,
		step=50,
		update=(lambda self, context: self.sync_prop('wl', context)),
	)

	rel_permittivity: bpy.props.FloatVectorProperty(
		name='Relative Permittivity',
		description='Represents a simple, complex permittivity',
		size=2,
		default=(1.0, 0.0),
		precision=2,
		update=(
			lambda self, context: self.sync_prop('rel_permittivity', context)
		),
	)

	####################
	# - Socket UI
	####################
	def draw_value(self, col: bpy.types.UILayout) -> None:
		col.prop(self, 'wl', text='λ')
		col.separator(factor=1.0)

		split = col.split(factor=0.35, align=False)

		col = split.column(align=True)
		col.label(text='ϵ_r (ℂ)')

		col = split.column(align=True)
		col.prop(self, 'rel_permittivity', text='')

	####################
	# - Computation of Default Value
	####################
	@property
	def value(self) -> td.Medium:
		freq = (
			spu.convert_to(
				VAC_SPEED_OF_LIGHT / (self.wl * self.unit),
				spu.hertz,
			)
			/ spu.hertz
		)
		return td.Medium.from_nk(
			n=self.rel_permittivity[0],
			k=self.rel_permittivity[1],
			freq=freq,
		)

	@value.setter
	def value(
		self, value: tuple[ConstrSympyExpr(allow_variables=False), complex]
	) -> None:
		_wl, rel_permittivity = value

		wl = float(
			spu.convert_to(
				_wl,
				self.unit,
			)
			/ self.unit
		)
		self.wl = wl
		self.rel_permittivity = (rel_permittivity.real, rel_permittivity.imag)

	def sync_unit_change(self):
		"""Override unit change to only alter frequency unit."""

		self.value = (
			self.wl * self.prev_unit,
			complex(*self.rel_permittivity),
		)
		self.prev_active_unit = self.active_unit


####################
# - Socket Configuration
####################
class MaxwellMediumSocketDef(pyd.BaseModel):
	socket_type: ct.SocketType = ct.SocketType.MaxwellMedium

	default_permittivity_real: float = 1.0
	default_permittivity_imag: float = 0.0

	def init(self, bl_socket: MaxwellMediumBLSocket) -> None:
		bl_socket.rel_permittivity = (
			self.default_permittivity_real,
			self.default_permittivity_imag,
		)


####################
# - Blender Registration
####################
BL_REGISTER = [
	MaxwellMediumBLSocket,
]
