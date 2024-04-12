import typing as typ

import bpy
import sympy as sp

from .....utils import logger
from ... import contracts as ct
from ... import sockets
from ...managed_objs.managed_bl_collection import preview_collection
from .. import base, events

log = logger.get(__name__)
console = logger.OUTPUT_CONSOLE


class ConsoleViewOperator(bpy.types.Operator):
	bl_idname = 'blender_maxwell.console_view_operator'
	bl_label = 'View Plots'

	@classmethod
	def poll(cls, context):
		return True

	def execute(self, context):
		node = context.node

		node.print_data_to_console()
		return {'FINISHED'}


class RefreshPlotViewOperator(bpy.types.Operator):
	bl_idname = 'blender_maxwell.refresh_plot_view_operator'
	bl_label = 'Refresh Plots'

	@classmethod
	def poll(cls, context):
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
		'Data': sockets.AnySocketDef(),
	}

	####################
	# - Properties
	####################
	auto_plot: bpy.props.BoolProperty(
		name='Auto-Plot',
		description='Whether to auto-plot anything plugged into the viewer node',
		default=False,
		update=lambda self, context: self.sync_prop('auto_plot', context),
	)

	auto_3d_preview: bpy.props.BoolProperty(
		name='Auto 3D Preview',
		description="Whether to auto-preview anything 3D, that's plugged into the viewer node",
		default=True,
		update=lambda self, context: self.sync_prop('auto_3d_preview', context),
	)

	####################
	# - UI
	####################
	def draw_operators(self, context, layout):
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
		row.prop(self, 'auto_plot', text='Plot', toggle=True)
		row.operator(
			RefreshPlotViewOperator.bl_idname,
			text='',
			icon='FILE_REFRESH',
		)

		## 3D Preview Options
		row = col.row(align=True)
		row.prop(self, 'auto_3d_preview', text='3D Preview', toggle=True)

	####################
	# - Methods
	####################
	def print_data_to_console(self):
		if not self.inputs['Data'].is_linked:
			return

		log.info('Printing Data to Console')
		data = self._compute_input('Data')
		if isinstance(data, sp.Basic):
			console.print(sp.pretty(data, use_unicode=True))
		else:
			console.print(data)

	####################
	# - Event Methods
	####################
	@events.on_value_changed(
		socket_name='Data',
		prop_name='auto_plot',
		props={'auto_plot'},
	)
	def on_changed_plot_preview(self, props):
		if self.inputs['Data'].is_linked and props['auto_plot']:
			self.trigger_action(ct.DataFlowAction.ShowPlot)

	@events.on_value_changed(
		socket_name='Data',
		prop_name='auto_3d_preview',
		props={'auto_3d_preview'},
	)
	def on_changed_3d_preview(self, props):
		node_tree = self.id_data

		# Remove Non-Repreviewed Previews on Close
		with node_tree.repreview_all():
			if self.inputs['Data'].is_linked and props['auto_3d_preview']:
				self.trigger_action(ct.DataFlowAction.ShowPreview)


####################
# - Blender Registration
####################
BL_REGISTER = [
	ConsoleViewOperator,
	RefreshPlotViewOperator,
	ViewerNode,
]
BL_NODES = {ct.NodeType.Viewer: (ct.NodeCategory.MAXWELLSIM_OUTPUTS)}
