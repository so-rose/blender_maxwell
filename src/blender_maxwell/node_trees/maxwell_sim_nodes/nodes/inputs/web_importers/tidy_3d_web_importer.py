import functools
import tempfile
from pathlib import Path
import typing as typ
from pathlib import Path

import bpy
import sympy as sp
import pydantic as pyd
import tidy3d as td
import tidy3d.web as td_web

from ......utils import tdcloud
from .... import contracts as ct
from .... import sockets
from ... import base

CACHE = {}


####################
# - Node
####################
class Tidy3DWebImporterNode(base.MaxwellSimNode):
	node_type = ct.NodeType.Tidy3DWebImporter
	bl_label = 'Tidy3DWebImporter'

	input_sockets = {
		'Cloud Task': sockets.Tidy3DCloudTaskSocketDef(
			should_exist=True,
		),
		'Cache Path': sockets.FilePathSocketDef(
			default_path=Path('loaded_simulation.hdf5')
		),
	}

	####################
	# - Output Methods
	####################
	@base.computes_output_socket(
		'FDTD Sim Data',
		input_sockets={'Cloud Task', 'Cache Path'},
	)
	def compute_fdtd_sim_data(self, input_sockets: dict) -> str:
		global CACHE
		if not CACHE.get(self.instance_id):
			CACHE[self.instance_id] = {'fdtd_sim_data': None}

		if CACHE[self.instance_id]['fdtd_sim_data'] is not None:
			return CACHE[self.instance_id]['fdtd_sim_data']

		if not (
			(cloud_task := input_sockets['Cloud Task']) is not None
			and isinstance(cloud_task, tdcloud.CloudTask)
			and cloud_task.status == 'success'
		):
			msg = "Won't attempt getting SimData"
			raise RuntimeError(msg)

		# Load the Simulation
		cache_path = input_sockets['Cache Path']
		if cache_path is None:
			print('CACHE PATH IS NONE WHY')
			return  ## I guess?
		if cache_path.is_file():
			sim_data = td.SimulationData.from_file(str(cache_path))

		else:
			sim_data = td_web.api.webapi.load(
				cloud_task.task_id,
				path=str(cache_path),
			)

		CACHE[self.instance_id]['fdtd_sim_data'] = sim_data
		return sim_data

	@base.computes_output_socket(
		'FDTD Sim',
		input_sockets={'Cloud Task'},
	)
	def compute_fdtd_sim(self, input_sockets: dict) -> str:
		if not isinstance(
			cloud_task := input_sockets['Cloud Task'], tdcloud.CloudTask
		):
			msg = 'Input cloud task does not exist'
			raise RuntimeError(msg)

		# Load the Simulation
		with tempfile.NamedTemporaryFile(delete=False) as f:
			_path_tmp = Path(f.name)
			_path_tmp.rename(f.name + '.json')
			path_tmp = Path(f.name + '.json')

		sim = td_web.api.webapi.load_simulation(
			cloud_task.task_id,
			path=str(path_tmp),
		)  ## TODO: Don't use td_web directly. Only through tdcloud
		Path(path_tmp).unlink()

		return sim

	####################
	# - Update
	####################
	@base.on_value_changed(
		socket_name='Cloud Task', input_sockets={'Cloud Task'}
	)
	def on_value_changed__cloud_task(self, input_sockets: dict):
		if (
			(cloud_task := input_sockets['Cloud Task']) is not None
			and isinstance(cloud_task, tdcloud.CloudTask)
			and cloud_task.status == 'success'
		):
			self.loose_output_sockets = {
				'FDTD Sim Data': sockets.MaxwellFDTDSimDataSocketDef(),
				'FDTD Sim': sockets.MaxwellFDTDSimSocketDef(),
			}
			return

		self.loose_output_sockets = {}

	@base.on_init()
	def on_init(self):
		self.on_value_changed__cloud_task()


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
