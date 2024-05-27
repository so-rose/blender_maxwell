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
import pydantic as pyd
import tidy3d as td

from blender_maxwell.utils import bl_cache, logger

from ... import contracts as ct
from .. import base

log = logger.get(__name__)


class MaxwellBoundCondBLSocket(base.MaxwellSimSocket):
	"""Describes a single of boundary condition to apply to the half-axis of a simulation domain.

	Attributes:
		default: The default boundary condition type.
	"""

	socket_type = ct.SocketType.MaxwellBoundCond
	bl_label = 'Maxwell Bound Cond'

	####################
	# - Properties
	####################
	default: ct.BoundCondType = bl_cache.BLField(ct.BoundCondType.Pml, prop_ui=True)

	# Capabilities
	## Allow a boundary condition compatible with any of the following axes.
	allow_axes: set[ct.SimSpaceAxis] = bl_cache.BLField(
		{ct.SimSpaceAxis.X, ct.SimSpaceAxis.Y, ct.SimSpaceAxis.Z},
	)
	## Present a boundary condition compatible with any of the following axes.
	present_axes: set[ct.SimSpaceAxis] = bl_cache.BLField(
		{ct.SimSpaceAxis.X, ct.SimSpaceAxis.Y, ct.SimSpaceAxis.Z},
	)

	####################
	# - UI
	####################
	def draw_value(self, col: bpy.types.UILayout) -> None:
		col.prop(self, self.blfields['default'], text='')

	####################
	# - FlowKind
	####################
	@bl_cache.cached_bl_property(
		depends_on={'active_kind', 'allow_axes', 'present_axes'}
	)
	def capabilities(self) -> ct.CapabilitiesFlow:
		return ct.CapabilitiesFlow(
			socket_type=self.socket_type,
			active_kind=self.active_kind,
			allow_any=self.allow_axes,
			present_any=self.present_axes,
		)

	@bl_cache.cached_bl_property(depends_on={'default'})
	def value(self) -> td.BoundaryEdge:
		return self.default.tidy3d_boundary_edge

	@value.setter
	def value(self, value: ct.BoundCondType) -> None:
		self.default = value


####################
# - Socket Configuration
####################
class MaxwellBoundCondSocketDef(base.SocketDef):
	socket_type: ct.SocketType = ct.SocketType.MaxwellBoundCond

	default: ct.BoundCondType = ct.BoundCondType.Pml
	allow_axes: set[ct.SimSpaceAxis] = pyd.Field(
		default={
			ct.SimSpaceAxis.X,
			ct.SimSpaceAxis.Y,
			ct.SimSpaceAxis.Z,
		}
	)
	present_axes: set[ct.SimSpaceAxis] = pyd.Field(
		default={
			ct.SimSpaceAxis.X,
			ct.SimSpaceAxis.Y,
			ct.SimSpaceAxis.Z,
		}
	)

	def init(self, bl_socket: MaxwellBoundCondBLSocket) -> None:
		bl_socket.default = self.default

		bl_socket.allow_axes = self.allow_axes
		bl_socket.present_axes = self.present_axes


####################
# - Blender Registration
####################
BL_REGISTER = [
	MaxwellBoundCondBLSocket,
]
