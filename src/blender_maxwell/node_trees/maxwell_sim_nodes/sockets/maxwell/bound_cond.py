import typing as typ

import bpy
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

	####################
	# - UI
	####################
	def draw_value(self, col: bpy.types.UILayout) -> None:
		col.prop(self, self.blfields['default'], text='')

	####################
	# - Computation of Default Value
	####################
	@property
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

	def init(self, bl_socket: MaxwellBoundCondBLSocket) -> None:
		bl_socket.default = self.default


####################
# - Blender Registration
####################
BL_REGISTER = [
	MaxwellBoundCondBLSocket,
]
