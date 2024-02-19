import typing as typ
import json
from pathlib import Path

import bpy
import sympy as sp
import pydantic as pyd

from .... import contracts
from .... import sockets
from ... import base

####################
# - Operators
####################
class JSONFileExporterPrintJSON(bpy.types.Operator):
	bl_idname = "blender_maxwell.json_file_exporter_print_json"
	bl_label = "Print the JSON of what's linked into a JSONFileExporterNode."

	@classmethod
	def poll(cls, context):
		return True

	def execute(self, context):
		node = context.node
		print(node.linked_data_as_json())
		return {'FINISHED'}

class JSONFileExporterSaveJSON(bpy.types.Operator):
	bl_idname = "blender_maxwell.json_file_exporter_save_json"
	bl_label = "Save the JSON of what's linked into a JSONFileExporterNode."

	@classmethod
	def poll(cls, context):
		return True

	def execute(self, context):
		node = context.node
		node.export_data_as_json()
		return {'FINISHED'}

####################
# - Node
####################
class JSONFileExporterNode(base.MaxwellSimTreeNode):
	node_type = contracts.NodeType.JSONFileExporter
	
	bl_label = "JSON File Exporter"
	#bl_icon = constants.ICON_SIM_INPUT
	
	input_sockets = {
		"json_path": sockets.FilePathSocketDef(
			label="JSON Path",
			default_path="simulation.json"
		),
		"data": sockets.AnySocketDef(
			label="Data",
		),
	}
	output_sockets = {}
	
	####################
	# - UI Layout
	####################
	def draw_operators(
		self,
		context: bpy.types.Context,
		layout: bpy.types.UILayout,
	) -> None:
		layout.operator(JSONFileExporterPrintJSON.bl_idname, text="Print")
		layout.operator(JSONFileExporterSaveJSON.bl_idname, text="Save")

	####################
	# - Methods
	####################
	def linked_data_as_json(self) -> str | None:
		if self.g_input_bl_socket("data").is_linked:
			data: typ.Any = self.compute_input("data")
			
			# Tidy3D Objects: Call .json()
			if hasattr(data, "json"):
				return data.json()
			
			# Pydantic Models: Call .model_dump_json()
			elif isinstance(data, pyd.BaseModel):
				return data.model_dump_json()
			
			# Finally: Try json.dumps (might fail)
			else:
				json.dumps(data)
	
	def export_data_as_json(self) -> None:
		if (data := self.linked_data_as_json()):
			data_dict = json.loads(data)
			with self.compute_input("json_path").open("w") as f:
				json.dump(data_dict, f, ensure_ascii=False, indent=4)


####################
# - Blender Registration
####################
BL_REGISTER = [
	JSONFileExporterPrintJSON,
	JSONFileExporterNode,
	JSONFileExporterSaveJSON,
]
BL_NODES = {
	contracts.NodeType.JSONFileExporter: (
		contracts.NodeCategory.MAXWELLSIM_OUTPUTS_EXPORTERS
	)
}
