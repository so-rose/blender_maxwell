import typing as typ
import json
from pathlib import Path

import bpy
import sympy as sp
import pydantic as pyd
import tidy3d as td

from .... import contracts as ct
from .... import sockets
from ... import base

####################
# - Operators
####################
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
class JSONFileExporterNode(base.MaxwellSimNode):
	node_type = ct.NodeType.JSONFileExporter
	
	bl_label = "JSON File Exporter"
	#bl_icon = constants.ICON_SIM_INPUT
	
	input_sockets = {
		"Data": sockets.AnySocketDef(),
		"JSON Path": sockets.FilePathSocketDef(
			default_path=Path("simulation.json")
		),
		"JSON Indent": sockets.IntegerNumberSocketDef(
			default_value=4,
		),
	}
	output_sockets = {
		"JSON String": sockets.TextSocketDef(),
	}
	
	####################
	# - UI Layout
	####################
	def draw_operators(
		self,
		context: bpy.types.Context,
		layout: bpy.types.UILayout,
	) -> None:
		layout.operator(JSONFileExporterSaveJSON.bl_idname, text="Save JSON")

	####################
	# - Methods
	####################
	def export_data_as_json(self) -> None:
		if (json_str := self.compute_output("JSON String")):
			data_dict = json.loads(json_str)
			with self._compute_input("JSON Path").open("w") as f:
				indent = self._compute_input("JSON Indent")
				json.dump(data_dict, f, ensure_ascii=False, indent=indent)
	
	####################
	# - Output Sockets
	####################
	@base.computes_output_socket(
		"JSON String",
		input_sockets={"Data"},
	)
	def compute_json_string(self, input_sockets: dict[str, typ.Any]) -> str | None:
		if not (data := input_sockets["Data"]):
			return None
		
		# Tidy3D Objects: Call .json()
		if hasattr(data, "json"):
			return data.json()
		
		# Pydantic Models: Call .model_dump_json()
		elif isinstance(data, pyd.BaseModel):
			return data.model_dump_json()
		
		else:
			json.dumps(data)


####################
# - Blender Registration
####################
BL_REGISTER = [
	JSONFileExporterSaveJSON,
	JSONFileExporterNode,
]
BL_NODES = {
	ct.NodeType.JSONFileExporter: (
		ct.NodeCategory.MAXWELLSIM_OUTPUTS_EXPORTERS
	)
}
