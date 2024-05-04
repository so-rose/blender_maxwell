# blender_maxwell
# Copyright (C) 2024 blender_maxwell Project Contributors
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import bpy
import sympy as sp
import sympy.physics.optics.polarization as spo_pol
import sympy.physics.units as spu

from blender_maxwell.utils import extra_sympy_units as spux

from ... import contracts as ct
from .. import base

StokesVector = spux.SympyExpr


class PhysicalPolBLSocket(base.MaxwellSimSocket):
	socket_type = ct.SocketType.PhysicalPol
	bl_label = 'Polarization'
	use_units = True

	def radianize(self, ang):
		return (
			spu.convert_to(
				ang * self.unit,
				spu.radian,
			)
			/ spu.radian
		)

	####################
	# - Properties
	####################
	model: bpy.props.EnumProperty(
		name='Polarization Model',
		description='A choice of polarization representation',
		items=[
			('UNPOL', 'Unpolarized', 'Unpolarized'),
			('LIN_ANG', 'Linear', 'Linearly polarized at angle'),
			('CIRC', 'Circular', 'Linearly polarized at angle'),
			('JONES', 'Jones', 'Polarized waves described by Jones vector'),
			('STOKES', 'Stokes', 'Linear x-pol of field'),
		],
		default='UNPOL',
		update=(lambda self, context: self.on_prop_changed('model', context)),
	)

	## Lin Ang
	lin_ang: bpy.props.FloatProperty(
		name='Pol. Angle',
		description='Angle to polarize linearly along',
		default=0.0,
		update=(lambda self, context: self.on_prop_changed('lin_ang', context)),
	)
	## Circ
	circ: bpy.props.EnumProperty(
		name='Pol. Orientation',
		description='LCP or RCP',
		items=[
			('LCP', 'LCP', "'Left Circular Polarization'"),
			('RCP', 'RCP', "'Right Circular Polarization'"),
		],
		default='LCP',
		update=(lambda self, context: self.on_prop_changed('circ', context)),
	)
	## Jones
	jones_psi: bpy.props.FloatProperty(
		name='Jones X-Axis Angle',
		description='Angle of the ellipse to the x-axis',
		default=0.0,
		precision=2,
		update=(lambda self, context: self.on_prop_changed('jones_psi', context)),
	)
	jones_chi: bpy.props.FloatProperty(
		name='Jones Major-Axis-Adjacent Angle',
		description='Angle of adjacent to the ellipse major axis',
		default=0.0,
		precision=2,
		update=(lambda self, context: self.on_prop_changed('jones_chi', context)),
	)

	## Stokes
	stokes_psi: bpy.props.FloatProperty(
		name='Stokes X-Axis Angle',
		description='Angle of the ellipse to the x-axis',
		default=0.0,
		precision=2,
		update=(lambda self, context: self.on_prop_changed('stokes_psi', context)),
	)
	stokes_chi: bpy.props.FloatProperty(
		name='Stokes Major-Axis-Adjacent Angle',
		description='Angle of adjacent to the ellipse major axis',
		default=0.0,
		precision=2,
		update=(lambda self, context: self.on_prop_changed('stokes_chi', context)),
	)
	stokes_p: bpy.props.FloatProperty(
		name='Stokes Polarization Degree',
		description='The degree of polarization',
		default=0.0,
		precision=2,
		update=(lambda self, context: self.on_prop_changed('stokes_p', context)),
	)
	stokes_I: bpy.props.FloatProperty(
		name='Stokes Field Intensity',
		description='The intensity of the polarized field',
		default=0.0,
		precision=2,
		update=(lambda self, context: self.on_prop_changed('stokes_I', context)),
	)  ## TODO: Units?

	####################
	# - UI
	####################
	def draw_value(self, col: bpy.types.UILayout) -> None:
		col.prop(self, 'model', text='')

		if self.model == 'LIN_ANG':
			col.prop(self, 'lin_ang', text='')

		elif self.model == 'CIRC':
			col.prop(self, 'circ', text='')

		elif self.model == 'JONES':
			split = col.split(factor=0.2, align=True)

			col = split.column(align=False)
			col.label(text='ψ,χ')

			col = split.row(align=True)
			col.prop(self, 'jones_psi', text='')
			col.prop(self, 'jones_chi', text='')

		elif self.model == 'STOKES':
			split = col.split(factor=0.2, align=True)
			# Split #1
			col = split.column(align=False)
			col.label(text='ψ,χ')
			col.label(text='p,I')

			# Split #2
			col = split.column(align=False)
			row = col.row(align=True)
			row.prop(self, 'stokes_psi', text='')
			row.prop(self, 'stokes_chi', text='')

			row = col.row(align=True)
			row.prop(self, 'stokes_p', text='')
			row.prop(self, 'stokes_I', text='')

			## TODO: Visualize stokes vector as oriented plane wave shape in plot (direct), in 3D (maybe in combination with a source), and on Poincare sphere.

	####################
	# - Default Value
	####################
	@property
	def _value_unpol(self) -> StokesVector:
		return spo_pol.stokes_vector(0, 0, 0)

	@property
	def _value_lin_ang(self) -> StokesVector:
		return spo_pol.stokes_vector(self.radianize(self.lin_ang), 0, 0)

	@property
	def _value_circ(self) -> StokesVector:
		return {
			'RCP': spo_pol.stokes_vector(0, sp.pi / 4, 0),
			'LCP': spo_pol.stokes_vector(0, -sp.pi / 4, 0),
		}[self.circ]

	@property
	def _value_jones(self) -> StokesVector:
		return spo_pol.jones_2_stokes(
			spo_pol.jones_vector(
				self.radianize(self.jones_psi),
				self.radianize(self.jones_chi),
			)
		)

	@property
	def _value_stokes(self) -> StokesVector:
		return spo_pol.stokes_vector(
			self.radianize(self.stokes_psi),
			self.radianize(self.stokes_chi),
			self.stokes_p,
			self.stokes_I,
		)

	@property
	def value(self) -> StokesVector:
		return {
			'UNPOL': self._value_unpol,
			'LIN_ANG': self._value_lin_ang,
			'CIRC': self._value_circ,
			'JONES': self._value_jones,
			'STOKES': self._value_stokes,
		}[self.model]

	def sync_unit_change(self) -> None:
		"""We don't have a setter, so we need to manually implement the unit change operation."""
		self.lin_ang = (
			spu.convert_to(
				self.lin_ang * self.prev_unit,
				self.unit,
			)
			/ self.unit
		)
		self.jones_psi = (
			spu.convert_to(
				self.jones_psi * self.prev_unit,
				self.unit,
			)
			/ self.unit
		)
		self.jones_chi = (
			spu.convert_to(
				self.jones_chi * self.prev_unit,
				self.unit,
			)
			/ self.unit
		)
		self.stokes_psi = (
			spu.convert_to(
				self.stokes_psi * self.prev_unit,
				self.unit,
			)
			/ self.unit
		)
		self.stokes_chi = (
			spu.convert_to(
				self.stokes_chi * self.prev_unit,
				self.unit,
			)
			/ self.unit
		)

		self.prev_active_unit = self.active_unit


####################
# - Socket Configuration
####################
class PhysicalPolSocketDef(base.SocketDef):
	socket_type: ct.SocketType = ct.SocketType.PhysicalPol

	def init(self, bl_socket: PhysicalPolBLSocket) -> None:
		pass


####################
# - Blender Registration
####################
BL_REGISTER = [
	PhysicalPolBLSocket,
]
