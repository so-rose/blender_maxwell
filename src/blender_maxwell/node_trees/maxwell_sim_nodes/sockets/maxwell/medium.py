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
import scipy as sc
import sympy.physics.units as spu
import tidy3d as td

from blender_maxwell.utils import extra_sympy_units as spux

from ... import contracts as ct
from .. import base

VAC_SPEED_OF_LIGHT = sc.constants.speed_of_light * spu.meter / spu.second

FIXED_WL = 500 * spu.nm


class MaxwellMediumBLSocket(base.MaxwellSimSocket):
	socket_type = ct.SocketType.MaxwellMedium
	bl_label = 'Maxwell Medium'

	####################
	# - Properties
	####################
	rel_permittivity: bpy.props.FloatVectorProperty(
		name='Relative Permittivity',
		description='Represents a simple, complex permittivity',
		size=2,
		default=(1.0, 0.0),
		precision=2,
		update=(
			lambda self, context: self.on_prop_changed('rel_permittivity', context)
		),
	)

	####################
	# - FlowKinds
	####################
	@property
	def value(self) -> td.Medium:
		freq = (
			spu.convert_to(
				VAC_SPEED_OF_LIGHT / FIXED_WL,
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
		self, value: tuple[spux.ConstrSympyExpr(allow_variables=False), complex]
	) -> None:
		rel_permittivity = value

		self.rel_permittivity = (rel_permittivity.real, rel_permittivity.imag)

	####################
	# - UI
	####################
	def draw_value(self, col: bpy.types.UILayout) -> None:
		split = col.split(factor=0.35, align=False)

		col = split.column(align=True)
		col.label(text='ϵ_r (ℂ)')

		col = split.column(align=True)
		col.prop(self, 'rel_permittivity', text='')


####################
# - Socket Configuration
####################
class MaxwellMediumSocketDef(base.SocketDef):
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
