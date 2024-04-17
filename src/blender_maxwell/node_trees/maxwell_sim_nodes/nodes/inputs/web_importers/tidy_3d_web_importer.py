import typing as typ
from pathlib import Path

from blender_maxwell.utils import logger

from ...... import info
from ......services import tdcloud
from .... import contracts as ct
from .... import sockets
from ... import base, events

log = logger.get(__name__)


def _sim_data_cache_path(task_id: str) -> Path:
	"""Compute an appropriate location for caching simulations downloaded from the internet, unique to each task ID.

	Arguments:
		task_id: The ID of the Tidy3D cloud task.
	"""
	(info.ADDON_CACHE / task_id).mkdir(exist_ok=True)
	return info.ADDON_CACHE / task_id / 'sim_data.hdf5'


####################
# - Node
####################
class Tidy3DWebImporterNode(base.MaxwellSimNode):
	node_type = ct.NodeType.Tidy3DWebImporter
	bl_label = 'Tidy3D Web Importer'

	input_sockets: typ.ClassVar = {
		'Cloud Task': sockets.Tidy3DCloudTaskSocketDef(
			should_exist=True,
		),
	}

	####################
	# - Event Methods
	####################
	@events.computes_output_socket(
		'FDTD Sim Data',
		input_sockets={'Cloud Task'},
	)
	def compute_sim_data(self, input_sockets: dict) -> str:
		## TODO: REMOVE TEST
		log.info('Loading SimulationData File')
		import sys

		for module_name, module in sys.modules.copy().items():
			if module_name == '__mp_main__':
				print('Problematic Module Entry', module_name)
				print(module)
				# print('MODULE REPR', module)
				continue
		# return td.SimulationData.from_file(
		# fname='/home/sofus/src/blender_maxwell/dev/sim_demo.hdf5'
		# )

		# Validate Task Availability
		if (cloud_task := input_sockets['Cloud Task']) is None:
			msg = f'"{self.bl_label}" CloudTask doesn\'t exist'
			raise RuntimeError(msg)

		# Validate Task Existence
		if not isinstance(cloud_task, tdcloud.CloudTask):
			msg = f'"{self.bl_label}" CloudTask input "{cloud_task}" has wrong "should_exists", as it isn\'t an instance of tdcloud.CloudTask'
			raise TypeError(msg)

		# Validate Task Status
		if cloud_task.status != 'success':
			msg = f'"{self.bl_label}" CloudTask is "{cloud_task.status}", not "success"'
			raise RuntimeError(msg)

		# Download and Return SimData
		return tdcloud.TidyCloudTasks.download_task_sim_data(
			cloud_task, _sim_data_cache_path(cloud_task.task_id)
		)

	@events.on_value_changed(
		socket_name='Cloud Task', run_on_init=True, input_sockets={'Cloud Task'}
	)
	def on_cloud_task_changed(self, input_sockets: dict):
		if (
			(cloud_task := input_sockets['Cloud Task']) is not None
			and isinstance(cloud_task, tdcloud.CloudTask)
			and cloud_task.status == 'success'
		):
			self.loose_output_sockets = {
				'FDTD Sim Data': sockets.MaxwellFDTDSimDataSocketDef(),
			}
		else:
			self.loose_output_sockets = {}


####################
# - Blender Registration
####################
BL_REGISTER = [
	Tidy3DWebImporterNode,
]
BL_NODES = {
	ct.NodeType.Tidy3DWebImporter: (ct.NodeCategory.MAXWELLSIM_INPUTS_WEBIMPORTERS)
}
