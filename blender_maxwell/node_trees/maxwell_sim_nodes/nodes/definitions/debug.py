import bpy
from .. import types, constants

class DebugPrinterNodeOperator(bpy.types.Operator):
	"""Print, to the console, the object retrieved by the calling
	DebugPrinterNode.
	"""
	
	bl_idname = "blender_maxwell.debug_printer_node_operator"
	bl_label = "Print the object linked into a DebugPrinterNode."

	@classmethod
	def poll(cls, context):
		return True

	def execute(self, context):
		node = context.node
		node.print_linked_data()
		return {'FINISHED'}

class DebugPrinterNode(types.MaxwellSimTreeNode):
	bl_idname = types.DebugPrinterNodeType
	bl_label = "Debug Printer"
	bl_icon = constants.tree_constants.ICON_SIM

	input_sockets = {
		"data": ("NodeSocketVirtual", "Data", lambda v: v),
	}
	output_sockets = {}
	
	####################
	# - Setup and Computation
	####################
	def print_linked_data(self):
		if self.inputs[self.input_sockets["data"][1]].is_linked:
			print(self.compute_input("data"))
		
	
	####################
	# - Node UI and Layout
	####################
	def draw_buttons(self, context, layout):
		layout.operator(DebugPrinterNodeOperator.bl_idname, text="Print Debug")



####################
# - Blender Registration
####################
BL_REGISTER = [
	DebugPrinterNodeOperator,
	DebugPrinterNode,
]
