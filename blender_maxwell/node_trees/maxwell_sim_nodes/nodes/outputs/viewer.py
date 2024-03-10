import functools
import typing as typ
import json
from pathlib import Path

import bpy
import sympy as sp
import pydantic as pyd
import tidy3d as td

from ... import contracts as ct
from ... import sockets
from .. import base


class ConsoleViewOperator(bpy.types.Operator):
	bl_idname = "blender_maxwell.console_view_operator"
	bl_label = "View Plots"

	@classmethod
	def poll(cls, context):
		return True

	def execute(self, context):
		node = context.node
		node.print_data_to_console()
		return {'FINISHED'}

class RefreshPlotViewOperator(bpy.types.Operator):
	bl_idname = "blender_maxwell.refresh_plot_view_operator"
	bl_label = "Refresh Plots"

	@classmethod
	def poll(cls, context):
		return True

	def execute(self, context):
		node = context.node
		node.trigger_action("value_changed", "Data")
		return {'FINISHED'}

####################
# - Node
####################
class ViewerNode(base.MaxwellSimNode):
	node_type = ct.NodeType.Viewer
	bl_label = "Viewer"
	
	input_sockets = {
		"Data": sockets.AnySocketDef(),
	}
	
	####################
	# - UI
	####################
	def draw_operators(self, context, layout):
		row = layout.row(align=True)
		row.label(text="Console")
		row.operator(ConsoleViewOperator.bl_idname, text="Print")
		
		row = layout.row(align=True)
		row.label(text="Plot")
		row.operator(RefreshPlotViewOperator.bl_idname, text="", icon="FILE_REFRESH")
	
	####################
	# - Methods
	####################
	def print_data_to_console(self):
		if not (data := self._compute_input("Data")):
			return
		
		if isinstance(data, sp.Basic):
			sp.pprint(data, use_unicode=True)
		
		print(str(data))
	
	####################
	# - Update
	####################
	@base.on_value_changed(socket_name="Data")
	def on_value_changed__data(self):
		self.trigger_action("show_plot")


####################
# - Blender Registration
####################
BL_REGISTER = [
	ConsoleViewOperator,
	RefreshPlotViewOperator,
	ViewerNode,
]
BL_NODES = {
	ct.NodeType.Viewer: (
		ct.NodeCategory.MAXWELLSIM_OUTPUTS
	)
}
