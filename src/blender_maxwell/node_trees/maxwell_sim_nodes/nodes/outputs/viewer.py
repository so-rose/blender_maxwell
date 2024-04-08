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
		print('Executing Operator')

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
		node.trigger_action('value_changed', 'Data')
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
		default=False,
		update=lambda self, context: self.sync_prop('auto_3d_preview', context),
	)

	cache__data_socket_linked: bpy.props.BoolProperty(
		name='Data Is Linked',
		description='Whether the Data input was linked last time it was checked.',
		default=True,
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
		props={'auto_plot'},
	)
	def on_changed_2d_data(self, props):
		# Show Plot
		## Don't have to un-show other plots.
		if self.inputs['Data'].is_linked and props['auto_plot']:
			self.trigger_action('show_plot')

	####################
	# - Event Methods: 3D Preview
	####################
	@events.on_value_changed(
		prop_name='auto_3d_preview',
		props={'auto_3d_preview'},
	)
	def on_changed_3d_preview(self, props):
		# Unpreview Everything
		node_tree = self.id_data
		node_tree.unpreview_all()

		# Trigger Preview Action
		if self.inputs['Data'].is_linked and props['auto_3d_preview']:
			log.info('Enabling 3D Previews from "%s"', self.name)
			self.trigger_action('show_preview')

	@events.on_value_changed(
		socket_name='Data',
	)
	def on_changed_3d_data(self):
		# Just Linked / Just Unlinked: Preview/Unpreview
		if self.inputs['Data'].is_linked ^ self.cache__data_socket_linked:
			self.on_changed_3d_preview()
			self.cache__data_socket_linked = self.inputs['Data'].is_linked


####################
# - Blender Registration
####################
BL_REGISTER = [
	ConsoleViewOperator,
	RefreshPlotViewOperator,
	ViewerNode,
]
BL_NODES = {ct.NodeType.Viewer: (ct.NodeCategory.MAXWELLSIM_OUTPUTS)}
