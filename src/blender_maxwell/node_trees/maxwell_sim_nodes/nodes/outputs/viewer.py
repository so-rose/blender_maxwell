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

from blender_maxwell.utils import bl_cache, logger
from blender_maxwell.utils import extra_sympy_units as spux

from ... import contracts as ct
from ... import sockets
from .. import base, events

log = logger.get(__name__)
console = logger.OUTPUT_CONSOLE


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
class ViewerNode(base.MaxwellSimNode):
	node_type = ct.NodeType.Viewer
	bl_label = 'Viewer'

	input_sockets: typ.ClassVar = {
		'Any': sockets.AnySocketDef(),
	}

	####################
	# - Properties
	####################
	print_kind: ct.FlowKind = bl_cache.BLField(ct.FlowKind.Value)
	auto_plot: bool = bl_cache.BLField(False)
	auto_3d_preview: bool = bl_cache.BLField(True)

	####################
	# - UI
	####################
	def draw_props(self, _: bpy.types.Context, layout: bpy.types.UILayout):
		layout.prop(self, self.blfields['print_kind'], text='')

	def draw_operators(self, _: bpy.types.Context, layout: bpy.types.UILayout):
		split = layout.split(factor=0.4)

		# Split LHS
		col = split.column(align=False)
		col.label(text='Console')
		col.label(text='Plot')
		col.label(text='3D')

		# Split RHS
		col = split.column(align=False)

		## Console Options
		col.operator(ConsoleViewOperator.bl_idname, text='Print')

		## Plot Options
		row = col.row(align=True)
		row.prop(self, self.blfields['auto_plot'], text='Plot', toggle=True)
		row.operator(
			RefreshPlotViewOperator.bl_idname,
			text='',
			icon='FILE_REFRESH',
		)

		## 3D Preview Options
		row = col.row(align=True)
		row.prop(self, self.blfields['auto_3d_preview'], text='3D Preview', toggle=True)

	####################
	# - Methods
	####################
	def print_data_to_console(self):
		if not self.inputs['Any'].is_linked:
			return

		log.info('Printing to Console')
		data = self._compute_input('Any', kind=self.print_kind, optional=True)

		if isinstance(data, spux.SympyType):
			console.print(sp.pretty(data, use_unicode=True))
		else:
			console.print(data)

	####################
	# - Event Methods
	####################
	@events.on_value_changed(
		socket_name='Any',
		prop_name='auto_plot',
		props={'auto_plot'},
	)
	def on_changed_plot_preview(self, props):
		node_tree = self.id_data

		# Unset Plot if Nothing Plotted
		with node_tree.replot():
			if props['auto_plot']:
				self.trigger_event(ct.FlowEvent.ShowPlot)

	@events.on_value_changed(
		socket_name='Any',
		prop_name='auto_3d_preview',
		props={'auto_3d_preview'},
	)
	def on_changed_3d_preview(self, props):
		node_tree = self.id_data

		# Remove Non-Repreviewed Previews on Close
		with node_tree.repreview_all():
			if props['auto_3d_preview']:
				self.trigger_event(ct.FlowEvent.ShowPreview)


####################
# - Blender Registration
####################
BL_REGISTER = [
	ConsoleViewOperator,
	RefreshPlotViewOperator,
	ViewerNode,
]
BL_NODES = {ct.NodeType.Viewer: (ct.NodeCategory.MAXWELLSIM_OUTPUTS)}
