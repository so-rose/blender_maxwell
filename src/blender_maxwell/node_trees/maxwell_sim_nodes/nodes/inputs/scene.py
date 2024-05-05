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

import enum
import typing as typ

import bpy
import sympy as sp

from blender_maxwell.utils import bl_cache, logger
from blender_maxwell.utils import extra_sympy_units as spux

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
			default_value=24,
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
	@property
	def scene_frame(self) -> int:
		"""Retrieve the current frame of the scene.

		Notes:
			A `frame_post` handler is registered that, on every frame change, causes a `DataChanged` to be emitted **with the property name `scene_frame`** (as if this were a normal property with an `update` method).
		"""
		return bpy.context.scene.frame_current

	@property
	def scene_frame_range(self) -> ct.LazyArrayRangeFlow:
		frame_start = bpy.context.scene.frame_start
		frame_stop = bpy.context.scene.frame_end
		return ct.LazyArrayRangeFlow(
			start=frame_start,
			stop=frame_stop,
			steps=frame_stop - frame_start + 1,
		)

	####################
	# - Property: Time Unit
	####################
	active_time_unit: enum.Enum = bl_cache.BLField(
		enum_cb=lambda self, _: self.search_units(), prop_ui=True
	)

	def search_units(self) -> list[ct.BLEnumElement]:
		return [
			(sp.sstr(unit), spux.sp_to_str(unit), sp.sstr(unit), '', i)
			for i, unit in enumerate(spux.PhysicalType.Time.valid_units)
		]

	@property
	def time_unit(self) -> spux.Unit | None:
		"""Gets the current active unit.

		Returns:
			The current active `sympy` unit.

			If the socket expression is unitless, this returns `None`.
		"""
		if self.active_time_unit is not None:
			return spux.unit_str_to_unit(self.active_time_unit)

		return None

	@time_unit.setter
	def time_unit(self, time_unit: spux.Unit | None) -> None:
		"""Set the unit, without touching the `raw_*` UI properties.

		Notes:
			To set a new unit, **and** convert the `raw_*` UI properties to the new unit, use `self.convert_unit()` instead.
		"""
		if time_unit in spux.PhysicalType.Time.valid_units:
			self.active_time_unit = sp.sstr(time_unit)
		else:
			msg = f'Tried to set invalid time unit {time_unit}'
			raise ValueError(msg)

	####################
	# - UI
	####################
	def draw_props(self, _: bpy.types.Context, col: bpy.types.UILayout) -> None:
		"""Draws the button that allows toggling between single and range output.

		Parameters:
			col: Target for defining UI elements.
		"""
		col.prop(self, self.blfields['active_time_unit'], toggle=True, text='Unit')

	####################
	# - FlowKinds
	####################
	@events.computes_output_socket(
		'Time',
		kind=ct.FlowKind.Value,
		input_sockets={'Frames / Unit'},
		props={'scene_frame', 'active_time_unit', 'time_unit'},
	)
	def compute_time(self, props, input_sockets) -> sp.Expr:
		return (
			props['scene_frame'] / input_sockets['Frames / Unit'] * props['time_unit']
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


@bpy.app.handlers.persistent
def update_scene_node_after_frame_changed(scene, depsgraph) -> None:
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
			node.trigger_event(ct.FlowEvent.DataChanged, prop_name='scene_frame')


bpy.app.handlers.frame_change_post.append(update_scene_node_after_frame_changed)

BL_NODES = {ct.NodeType.Scene: (ct.NodeCategory.MAXWELLSIM_INPUTS)}
