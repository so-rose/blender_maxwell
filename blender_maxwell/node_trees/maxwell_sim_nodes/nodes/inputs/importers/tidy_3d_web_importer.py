import functools
import tempfile
from pathlib import Path
import typing as typ
from pathlib import Path

import bpy
import sympy as sp
import pydantic as pyd
import tidy3d as td
import tidy3d.web as _td_web

from ......utils.auth_td_web import g_td_web, is_td_web_authed
from .... import contracts as ct
from .... import sockets
from ... import base

@functools.cache
def task_status(task_id: str):
	task = _td_web.api.webapi.get_info(task_id)
	return task.status

####################
# - Node
####################
class Tidy3DWebImporterNode(base.MaxwellSimNode):
	node_type = ct.NodeType.Tidy3DWebImporter
	bl_label = "Tidy3DWebImporter"
	
	input_sockets = {
		"Cloud Task": sockets.Tidy3DCloudTaskSocketDef(
			task_exists=True,
		),
	}
	output_sockets = {}
	
	####################
	# - UI
	####################
	def draw_info(self, context, layout): pass
	
	####################
	# - Output Methods
	####################
	@base.computes_output_socket(
		"FDTD Sim",
		input_sockets={"Cloud Task"},
	)
	def compute_cloud_task(self, input_sockets: dict) -> str:
		if not isinstance(task_id := input_sockets["Cloud Task"], str):
			msg ="Input task does not exist" 
			raise ValueError(msg)
		
		# Load the Simulation
		td_web = g_td_web(None)  ## Presume already auth'ed
		with tempfile.NamedTemporaryFile(delete=False) as f:
			_path_tmp = Path(f.name)
			_path_tmp.rename(f.name + ".json")
			path_tmp = Path(f.name + ".json")
		
		cloud_sim = _td_web.api.webapi.load_simulation(
			task_id,
			path=str(path_tmp),
		)
		Path(path_tmp).unlink()
		
		return cloud_sim
	
	####################
	# - Update
	####################
	@base.on_value_changed(
		socket_name="Cloud Task",
		input_sockets={"Cloud Task"}
	)
	def on_value_changed__cloud_task(self, input_sockets: dict):
		task_status.cache_clear()
		if (
			(task_id := input_sockets["Cloud Task"]) is None
			or isinstance(task_id, dict)
			or task_status(task_id) != "success"
			or not is_td_web_authed
		):
			if self.loose_output_sockets: self.loose_output_sockets = {}
			return
		
		td_web = g_td_web(None)  ## Presume already auth'ed
		
		self.loose_output_sockets = {
			"FDTD Sim": sockets.MaxwellFDTDSimSocketDef(),
			"FDTD Sim Data": sockets.AnySocketDef(),
		}


####################
# - Blender Registration
####################
BL_REGISTER = [
	Tidy3DWebImporterNode,
]
BL_NODES = {
	ct.NodeType.Tidy3DWebImporter: (
		ct.NodeCategory.MAXWELLSIM_INPUTS_IMPORTERS
	)
}
