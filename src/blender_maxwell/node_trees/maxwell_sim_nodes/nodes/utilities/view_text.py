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

import typing as typ

import bpy
import sympy as sp
import tidy3d as td

from blender_maxwell.utils import bl_cache, logger
from blender_maxwell.utils import sympy_extra as spux

from ... import contracts as ct
from ... import sockets
from .. import base, events

log = logger.get(__name__)

FK = ct.FlowKind
FS = ct.FlowSignal


class ConsoleViewOperator(bpy.types.Operator):
	bl_idname = 'blender_maxwell.console_view_operator'
	bl_label = 'View Plots'

	@classmethod
	def poll(cls, _: bpy.types.Context):
		return True

	def execute(self, context):
		node = context.node

		node.print_data_to_console()
		return {'FINISHED'}


class RefreshPlotViewOperator(bpy.types.Operator):
	bl_idname = 'blender_maxwell.refresh_plot_view_operator'
	bl_label = 'Refresh Plots'

	@classmethod
	def poll(cls, _: bpy.types.Context):
		return True

	def execute(self, context):
		node = context.node
		node.on_changed_plot_preview()
		return {'FINISHED'}


####################
# - Node
####################
class ViewTextNode(base.MaxwellSimNode):
	node_type = ct.NodeType.ViewText
	bl_label = 'View Text'
	# use_sim_node_name = True

	input_sockets: typ.ClassVar = {
		'Text': sockets.StringSocketDef(),
	}

	####################
	# - Properties
	####################
	push_live: bool = bl_cache.BLField(True)

	####################
	# - Properties: Computed FlowKinds
	####################
	@events.on_value_changed(
		socket_name={'Text': FK.Value},
		prop_name={'push_live', 'sim_node_name'},
		# Loaded
		inscks_kinds={'Text': FK.Value},
		input_sockets_optional={'Text'},
		props={'push_live', 'sim_node_name'},
	)
	def on_text_changed(self, props, input_sockets) -> None:
		sim_node_name = props['sim_node_name']
		push_live = props['push_live']

		if push_live:
			if bpy.data.texts.get(sim_node_name) is None:
				bpy.data.texts.new(sim_node_name)
			bl_text = bpy.data.texts[sim_node_name]

			bl_text.clear()

			text = input_sockets['Text']
			has_text = not FS.check(text)
			if has_text:
				bl_text.write(text)

	####################
	# - UI
	####################
	def draw_props(self, _: bpy.types.Context, layout: bpy.types.UILayout):
		row = layout.row(align=True)

		row.prop(self, self.blfields['push_live'], text='Write Live', toggle=True)

	def draw_info(self, _: bpy.types.Context, layout: bpy.types.UILayout):
		pass


####################
# - Blender Registration
####################
BL_REGISTER = [
	ViewTextNode,
]
BL_NODES = {ct.NodeType.ViewText: (ct.NodeCategory.MAXWELLSIM_UTILITIES)}
