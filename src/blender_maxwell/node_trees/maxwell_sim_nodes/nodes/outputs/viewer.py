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
	auto_expr: bool = bl_cache.BLField(True)
	debug_mode: bool = bl_cache.BLField(False)

	# Debug Mode
	console_print_kind: ct.FlowKind = bl_cache.BLField(ct.FlowKind.Value)
	auto_plot: bool = bl_cache.BLField(True)
	auto_3d_preview: bool = bl_cache.BLField(True)

	####################
	# - Properties: Computed FlowKinds
	####################
	@events.on_value_changed(
		socket_name='Any',
	)
	def on_input_changed(self) -> None:
		self.input_flow = bl_cache.Signal.InvalidateCache

	@bl_cache.cached_bl_property()
	def input_flow(self) -> dict[ct.FlowKind, typ.Any | None]:
		input_flow = {}

		for flow_kind in list(ct.FlowKind):
			flow = self._compute_input('Any', kind=flow_kind)
			has_flow = not ct.FlowSignal.check(flow)

			if has_flow:
				input_flow |= {flow_kind: flow}
			else:
				input_flow |= {flow_kind: None}

		return input_flow

	####################
	# - Property: Input Expression String Lines
	####################
	@bl_cache.cached_bl_property(depends_on={'input_flow'})
	def input_expr_str_entries(self) -> list[list[str]] | None:
		value = self.input_flow.get(ct.FlowKind.Value)

		def sp_pretty(v: spux.SympyExpr) -> spux.SympyExpr:
			## sp.pretty makes new lines and wreaks havoc.
			return spux.sp_to_str(v.n(4))

		if isinstance(value, spux.SympyType):
			if isinstance(value, sp.MatrixBase):
				return [
					[sp_pretty(value[row, col]) for col in range(value.shape[1])]
					for row in range(value.shape[0])
				]

			return [[sp_pretty(value)]]
		return None

	####################
	# - UI
	####################
	def draw_props(self, _: bpy.types.Context, layout: bpy.types.UILayout):
		row = layout.row(align=True)

		# Debug Mode On/Off
		row.prop(self, self.blfields['debug_mode'], text='Debug', toggle=True)

		# Automatic Expression Printing
		row.prop(self, self.blfields['auto_expr'], text='Expr', toggle=True)

		# Debug Mode Operators
		if self.debug_mode:
			layout.prop(self, self.blfields['console_print_kind'], text='')

	def draw_operators(self, _: bpy.types.Context, layout: bpy.types.UILayout):
		# Live Expression
		if self.debug_mode:
			layout.operator(ConsoleViewOperator.bl_idname, text='Console Print')

			split = layout.split(factor=0.4)

			# Split LHS
			col = split.column(align=False)
			col.label(text='Plot')
			col.label(text='3D')

			# Split RHS
			col = split.column(align=False)

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
			row.prop(
				self, self.blfields['auto_3d_preview'], text='3D Preview', toggle=True
			)

	def draw_info(self, _: bpy.types.Context, layout: bpy.types.UILayout):
		# Live Expression
		if self.auto_expr and self.input_expr_str_entries is not None:
			box = layout.box()

			expr_rows = len(self.input_expr_str_entries)
			expr_cols = len(self.input_expr_str_entries[0])
			shape_str = (
				f'({expr_rows}×{expr_cols})'
				if expr_rows != 1 or expr_cols != 1
				else '(Scalar)'
			)

			row = box.row()
			row.alignment = 'CENTER'
			row.label(text=f'Expr {shape_str}')

			if (
				len(self.input_expr_str_entries) == 1
				and len(self.input_expr_str_entries[0]) == 1
			):
				row = box.row()
				row.alignment = 'CENTER'
				row.label(text=self.input_expr_str_entries[0][0])
			else:
				grid = box.grid_flow(
					row_major=True,
					columns=len(self.input_expr_str_entries[0]),
					align=True,
				)
				for row in self.input_expr_str_entries:
					for entry in row:
						grid.label(text=entry)

	####################
	# - Methods
	####################
	def print_data_to_console(self):
		if not self.inputs['Any'].is_linked:
			return

		log.info('Printing to Console')
		data = self._compute_input('Any', kind=self.console_print_kind, optional=True)

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
			if props['auto_plot'] and self.inputs['Any'].is_linked:
				self.inputs['Any'].links[0].from_socket.node.trigger_event(
					ct.FlowEvent.ShowPlot
				)

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
