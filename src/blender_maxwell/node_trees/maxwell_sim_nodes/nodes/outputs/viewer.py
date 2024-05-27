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
		"""Lightweight invalidator, which invalidates the more specific `cached_bl_property` used to determine when something ex. plot-related has changed.

		Calls `get_flow`, which will be called again when regenerating the `cached_bl_property`s.
		This **does not** call the flow twice, as `self._compute_input()` will be cached the first time.
		"""
		for flow_kind in list(ct.FlowKind):
			flow = self.get_flow(
				flow_kind, always_load=flow_kind is ct.FlowKind.Previews
			)
			if flow is not None:
				setattr(
					self,
					'input_' + flow_kind.property_name,
					bl_cache.Signal.InvalidateCache,
				)

	@bl_cache.cached_bl_property(depends_on={'auto_expr'})
	def input_capabilities(self) -> ct.CapabilitiesFlow | None:
		return self.get_flow(ct.FlowKind.Capabilities)

	@bl_cache.cached_bl_property(depends_on={'auto_expr'})
	def input_previews(self) -> ct.PreviewsFlow | None:
		return self.get_flow(ct.FlowKind.Previews, always_load=True)

	@bl_cache.cached_bl_property(depends_on={'auto_expr'})
	def input_value(self) -> ct.ValueFlow | None:
		return self.get_flow(ct.FlowKind.Value)

	@bl_cache.cached_bl_property(depends_on={'auto_expr'})
	def input_array(self) -> ct.ArrayFlow | None:
		return self.get_flow(ct.FlowKind.Array)

	@bl_cache.cached_bl_property(depends_on={'auto_expr'})
	def input_lazy_range(self) -> ct.RangeFlow | None:
		return self.get_flow(ct.FlowKind.Range)

	@bl_cache.cached_bl_property(depends_on={'auto_expr'})
	def input_lazy_func(self) -> ct.FuncFlow | None:
		return self.get_flow(ct.FlowKind.Func)

	@bl_cache.cached_bl_property(depends_on={'auto_expr'})
	def input_params(self) -> ct.ParamsFlow | None:
		return self.get_flow(ct.FlowKind.Params)

	@bl_cache.cached_bl_property(depends_on={'auto_expr'})
	def input_info(self) -> ct.InfoFlow | None:
		return self.get_flow(ct.FlowKind.Info)

	def get_flow(
		self, flow_kind: ct.FlowKind, always_load: bool = False
	) -> typ.Any | None:
		"""Generic interface to simplify getting `FlowKind` properties on the viewer node."""
		if self.auto_expr or always_load:
			flow = self._compute_input('Any', kind=flow_kind)
			has_flow = not ct.FlowSignal.check(flow)

			if has_flow:
				return flow
			return None
		return None

	####################
	# - Property: Input Expression String Lines
	####################
	@bl_cache.cached_bl_property(depends_on={'input_value'})
	def input_expr_str_entries(self) -> list[list[str]] | None:
		value = self.input_value
		if value is None:
			return None

		# Parse SympyType
		def sp_pretty(v: spux.SympyExpr) -> spux.SympyExpr:
			## -> The real sp.pretty makes new lines and wreaks havoc.
			return spux.sp_to_str(v.n(4))

		if isinstance(value, spux.SympyType):
			if isinstance(value, sp.MatrixBase):
				return [
					[sp_pretty(value[row, col]) for col in range(value.shape[1])]
					for row in range(value.shape[0])
				]

			return [[sp_pretty(value)]]

		# Parse Tidy3D Types
		if isinstance(value, td.Structure):
			return [
				[str(key), str(value)]
				for key, value in dict(value).items()
				if key not in ['type', 'geometry', 'medium']
			] + [
				[str(key), str(value)]
				for key, value in dict(value.geometry).items()
				if key != 'type'
			]
		if isinstance(value, td.components.base.Tidy3dBaseModel):
			return [
				[str(key), str(value)]
				for key, value in dict(value).items()
				if key != 'type'
			]

		return None

	####################
	# - UI
	####################
	def draw_props(self, _: bpy.types.Context, layout: bpy.types.UILayout):
		row = layout.row(align=True)

		# Automatic Expression Printing
		row.prop(self, self.blfields['auto_expr'], text='Live', toggle=True)

		# Debug Mode On/Off
		row.prop(self, self.blfields['debug_mode'], text='Debug', toggle=True)

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
		flow = self._compute_input('Any', kind=self.console_print_kind)

		log.info('Printing to Console')
		if isinstance(flow, spux.SympyType):
			console.print(sp.pretty(flow, use_unicode=True))
		else:
			console.print(flow)

	####################
	# - Event Methods
	####################
	@events.on_value_changed(
		# Trigger
		prop_name={'input_previews', 'auto_plot'},
		# Loaded
		props={'input_previews', 'auto_plot'},
	)
	def on_changed_plot_preview(self, props):
		previews = props['input_previews']
		if previews is not None:
			if props['auto_plot']:
				bl_socket = self.inputs['Any']
				if bl_socket.is_linked:
					bl_socket.links[0].from_node.compute_plot()

			previews.update_image_preview()
		else:
			ct.PreviewsFlow.hide_image_preview()

	@events.on_value_changed(
		# Trigger
		prop_name={'input_previews', 'auto_3d_preview'},
		# Loaded
		props={'input_previews', 'auto_3d_preview'},
	)
	def on_changed_3d_preview(self, props):
		previews = props['input_previews']
		if previews is not None and props['auto_3d_preview']:
			previews.update_bl_object_previews()
		else:
			ct.PreviewsFlow.hide_bl_object_previews()


####################
# - Blender Registration
####################
BL_REGISTER = [
	ConsoleViewOperator,
	RefreshPlotViewOperator,
	ViewerNode,
]
BL_NODES = {ct.NodeType.Viewer: (ct.NodeCategory.MAXWELLSIM_OUTPUTS)}
