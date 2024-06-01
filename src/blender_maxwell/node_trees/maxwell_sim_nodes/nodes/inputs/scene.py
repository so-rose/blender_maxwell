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

"""Implements `SceneNode`."""

import typing as typ

import bpy
import sympy as sp
import sympy.physics.units as spu

from blender_maxwell.utils import bl_cache, logger
from blender_maxwell.utils import sympy_extra as spux

from ... import contracts as ct
from ... import sockets
from .. import base, events

log = logger.get(__name__)


class SceneNode(base.MaxwellSimNode):
	"""Gathers data from the Blender scene for use in the node tree.

	Attributes:
		use_range: Whether to specify a range of wavelengths/frequencies, or just one.
	"""

	node_type = ct.NodeType.Scene
	bl_label = 'Scene'

	input_sockets: typ.ClassVar = {
		'Frames / Unit': sockets.ExprSocketDef(
			mathtype=spux.MathType.Integer,
			default_value=48,
		),
		'Unit': sockets.ExprSocketDef(
			default_unit=spu.ps,
			default_value=1,
		),
	}
	output_sockets: typ.ClassVar = {
		'Time': sockets.ExprSocketDef(
			physical_type=spux.PhysicalType.Time,
		),
		'Frame': sockets.ExprSocketDef(
			mathtype=spux.MathType.Integer,
		),
	}

	####################
	# - Properties: Frame
	####################
	@bl_cache.cached_bl_property()
	def scene_frame(self) -> int:
		"""Retrieve the current frame of the scene.

		Notes:
			A `frame_post` handler is registered that, on every frame change, causes a `DataChanged` to be emitted **with the property name `scene_frame`** (as if this were a normal property with an `update` method).
		"""
		return bpy.context.scene.frame_current

	@property
	def scene_frame_range(self) -> ct.RangeFlow:
		"""Retrieve the current start/end frame of the scene, with `steps` corresponding to single-frame steps."""
		frame_start = bpy.context.scene.frame_start
		frame_stop = bpy.context.scene.frame_end
		return ct.RangeFlow(
			start=frame_start,
			stop=frame_stop,
			steps=frame_stop - frame_start + 1,
		)

	####################
	# - FlowKinds
	####################
	@events.computes_output_socket(
		'Time',
		kind=ct.FlowKind.Value,
		input_sockets={'Frames / Unit', 'Unit'},
		props={'scene_frame'},
	)
	def compute_time(self, props, input_sockets) -> sp.Expr:
		return (
			props['scene_frame']
			/ input_sockets['Frames / Unit']
			* input_sockets['Unit']
		)

	@events.computes_output_socket(
		'Frame',
		kind=ct.FlowKind.Value,
		props={'scene_frame'},
	)
	def compute_frame(self, props) -> sp.Expr:
		return props['scene_frame']


####################
# - Blender Registration
####################
BL_REGISTER = [
	SceneNode,
]
BL_NODES = {ct.NodeType.Scene: (ct.NodeCategory.MAXWELLSIM_INPUTS)}


####################
# - Blender Handlers
####################
@bpy.app.handlers.persistent
def update_scene_node_after_frame_changed(
	scene: bpy.types.Scene,  # noqa: ARG001
	depsgraph: bpy.types.Depsgraph,  # noqa: ARG001
) -> None:
	"""Invalidate the cached scene frame on all `SceneNode`s in all active simulation node trees, whenever the frame changes."""
	for node_tree in [
		_node_tree
		for _node_tree in bpy.data.node_groups
		if _node_tree.bl_idname == ct.TreeType.MaxwellSim.value and _node_tree.is_active
	]:
		for node in [
			_node
			for _node in node_tree.nodes
			if hasattr(_node, 'node_type') and _node.node_type == ct.NodeType.Scene
		]:
			node.scene_frame = bl_cache.Signal.InvalidateCache


bpy.app.handlers.frame_change_post.append(update_scene_node_after_frame_changed)
