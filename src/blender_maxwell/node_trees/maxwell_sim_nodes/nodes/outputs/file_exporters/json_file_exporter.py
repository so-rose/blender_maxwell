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

import json
import typing as typ
from pathlib import Path

import bpy
import pydantic as pyd

from .... import contracts as ct
from .... import sockets
from ... import base, events


####################
# - Operators
####################
class JSONFileExporterSaveJSON(bpy.types.Operator):
	bl_idname = 'blender_maxwell.json_file_exporter_save_json'
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

	bl_label = 'JSON File Exporter'
	# bl_icon = constants.ICON_SIM_INPUT

	input_sockets = {
		'Data': sockets.AnySocketDef(),
		'JSON Path': sockets.FilePathSocketDef(default_path=Path('simulation.json')),
		'JSON Indent': sockets.IntegerNumberSocketDef(
			default_value=4,
		),
	}
	output_sockets = {
		'JSON String': sockets.StringSocketDef(),
	}

	####################
	# - UI Layout
	####################
	def draw_operators(
		self,
		context: bpy.types.Context,
		layout: bpy.types.UILayout,
	) -> None:
		layout.operator(JSONFileExporterSaveJSON.bl_idname, text='Save JSON')

	####################
	# - Methods
	####################
	def export_data_as_json(self) -> None:
		if json_str := self.compute_output('JSON String'):
			data_dict = json.loads(json_str)
			with self._compute_input('JSON Path').open('w') as f:
				indent = self._compute_input('JSON Indent')
				json.dump(data_dict, f, ensure_ascii=False, indent=indent)

	####################
	# - Output Sockets
	####################
	@events.computes_output_socket(
		'JSON String',
		input_sockets={'Data'},
	)
	def compute_json_string(self, input_sockets: dict[str, typ.Any]) -> str | None:
		if not (data := input_sockets['Data']):
			return None

		# Tidy3D Objects: Call .json()
		if hasattr(data, 'json'):
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
	ct.NodeType.JSONFileExporter: (ct.NodeCategory.MAXWELLSIM_OUTPUTS_FILEEXPORTERS)
}
