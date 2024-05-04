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
import tidy3d as td

from ... import contracts as ct
from .. import base


class MaxwellSimGridBLSocket(base.MaxwellSimSocket):
	socket_type = ct.SocketType.MaxwellSimGrid
	bl_label = 'Maxwell Sim Grid'

	####################
	# - Properties
	####################
	min_steps_per_wl: bpy.props.FloatProperty(
		name='Minimum Steps per Wavelength',
		description='How many grid steps to ensure per wavelength',
		default=10.0,
		min=0.01,
		# step=10,
		precision=2,
		update=(
			lambda self, context: self.on_prop_changed('min_steps_per_wl', context)
		),
	)

	####################
	# - Socket UI
	####################
	def draw_value(self, col: bpy.types.UILayout) -> None:
		split = col.split(factor=0.5, align=False)

		col = split.column(align=True)
		col.label(text='min. stp/λ')

		col = split.column(align=True)
		col.prop(self, 'min_steps_per_wl', text='')

	####################
	# - Computation of Default Value
	####################
	@property
	def value(self) -> td.GridSpec:
		return td.GridSpec.auto(
			min_steps_per_wvl=self.min_steps_per_wl,
		)


####################
# - Socket Configuration
####################
class MaxwellSimGridSocketDef(base.SocketDef):
	socket_type: ct.SocketType = ct.SocketType.MaxwellSimGrid

	min_steps_per_wl: float = 10.0

	def init(self, bl_socket: MaxwellSimGridBLSocket) -> None:
		bl_socket.min_steps_per_wl = self.min_steps_per_wl


####################
# - Blender Registration
####################
BL_REGISTER = [
	MaxwellSimGridBLSocket,
]
