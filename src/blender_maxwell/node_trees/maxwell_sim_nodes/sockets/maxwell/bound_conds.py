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

"""Implements the `MaxwellBoundCondsBLSocket` socket."""

import bpy
import tidy3d as td

from blender_maxwell.utils import bl_cache, logger

from ... import contracts as ct
from .. import base

log = logger.get(__name__)


class MaxwellBoundCondsBLSocket(base.MaxwellSimSocket):
	"""Describes a set of boundary conditions to apply to a simulation domain.

	Attributes:
		show_definition: Toggle to show/hide default per-axis boundary conditions.
		x_pos: Default boundary condition to apply at the boundary of the sim domain's positive x-axis.
		x_neg: Default boundary condition to apply at the boundary of the sim domain's negative x-axis.
		y_pos: Default boundary condition to apply at the boundary of the sim domain's positive y-axis.
		y_neg: Default boundary condition to apply at the boundary of the sim domain's negative y-axis.
		z_pos: Default boundary condition to apply at the boundary of the sim domain's positive z-axis.
		z_neg: Default boundary condition to apply at the boundary of the sim domain's negative z-axis.
	"""

	socket_type = ct.SocketType.MaxwellBoundConds
	bl_label = 'Maxwell Bound Box'

	####################
	# - Properties
	####################
	show_definition: bool = bl_cache.BLField(False, prop_ui=True)

	x_pos: ct.BoundCondType = bl_cache.BLField(ct.BoundCondType.Pml, prop_ui=True)
	x_neg: ct.BoundCondType = bl_cache.BLField(ct.BoundCondType.Pml, prop_ui=True)
	y_pos: ct.BoundCondType = bl_cache.BLField(ct.BoundCondType.Pml, prop_ui=True)
	y_neg: ct.BoundCondType = bl_cache.BLField(ct.BoundCondType.Pml, prop_ui=True)
	z_pos: ct.BoundCondType = bl_cache.BLField(ct.BoundCondType.Pml, prop_ui=True)
	z_neg: ct.BoundCondType = bl_cache.BLField(ct.BoundCondType.Pml, prop_ui=True)

	####################
	# - UI
	####################
	def draw_label_row(self, row: bpy.types.UILayout, text) -> None:
		row.label(text=text)
		row.prop(
			self,
			self.blfields['show_definition'],
			toggle=True,
			text='',
			icon=ct.Icon.ToggleSocketInfo,
		)

	def draw_value(self, col: bpy.types.UILayout) -> None:
		if self.show_definition:
			for axis in ['x', 'y', 'z']:
				row = col.row(align=False)
				split = row.split(factor=0.2, align=False)

				_col = split.column(align=True)
				_col.alignment = 'RIGHT'
				_col.label(text=axis + ' -')
				_col.label(text=' +')

				_col = split.column(align=True)
				_col.prop(self, self.blfields[axis + '_neg'], text='')
				_col.prop(self, self.blfields[axis + '_pos'], text='')

	####################
	# - Computation of Default Value
	####################
	@property
	def value(self) -> td.BoundarySpec:
		"""Compute a user-defined default value for simulation boundary conditions, from certain common/sensible options.

		Each half-axis has a selection pulled from `ct.BoundCondType`.

		Returns:
			A usable `tidy3d` boundary specification.
		"""
		log.debug(
			'%s|%s: Computing default value for Boundary Conditions',
			self.node.sim_node_name,
			self.bl_label,
		)
		return td.BoundarySpec(
			x=td.Boundary(
				plus=self.x_pos.tidy3d_boundary_edge,
				minus=self.x_neg.tidy3d_boundary_edge,
			),
			y=td.Boundary(
				plus=self.y_pos.tidy3d_boundary_edge,
				minus=self.y_neg.tidy3d_boundary_edge,
			),
			z=td.Boundary(
				plus=self.z_pos.tidy3d_boundary_edge,
				minus=self.z_neg.tidy3d_boundary_edge,
			),
		)


####################
# - Socket Configuration
####################
class MaxwellBoundCondsSocketDef(base.SocketDef):
	socket_type: ct.SocketType = ct.SocketType.MaxwellBoundConds

	default_x_pos: ct.BoundCondType = ct.BoundCondType.Pml
	default_x_neg: ct.BoundCondType = ct.BoundCondType.Pml
	default_y_pos: ct.BoundCondType = ct.BoundCondType.Pml
	default_y_neg: ct.BoundCondType = ct.BoundCondType.Pml
	default_z_pos: ct.BoundCondType = ct.BoundCondType.Pml
	default_z_neg: ct.BoundCondType = ct.BoundCondType.Pml

	def init(self, bl_socket: MaxwellBoundCondsBLSocket) -> None:
		bl_socket.x_pos = self.default_x_pos
		bl_socket.x_neg = self.default_x_neg
		bl_socket.y_pos = self.default_y_pos
		bl_socket.y_neg = self.default_y_neg
		bl_socket.z_pos = self.default_z_pos
		bl_socket.z_neg = self.default_z_neg


####################
# - Blender Registration
####################
BL_REGISTER = [
	MaxwellBoundCondsBLSocket,
]
