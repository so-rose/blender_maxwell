import typing as typ
from pathlib import Path

import bpy
import tidy3d as td

from blender_maxwell.services import tdcloud
from blender_maxwell.utils import bl_cache, logger

from .... import contracts as ct
from .... import sockets
from ... import base, events

log = logger.get(__name__)


class LoadCloudSim(bpy.types.Operator):
	bl_idname = ct.OperatorType.NodeLoadCloudSim
	bl_label = '(Re)Load Sim'
	bl_description = '(Re)Load simulation data associated with the attached cloud task'

	@classmethod
	def poll(cls, context):
		return (
			# Node Type
			hasattr(context, 'node')
			and hasattr(context.node, 'node_type')
			and context.node.node_type == ct.NodeType.Tidy3DWebImporter
			# Cloud Status
			and tdcloud.IS_ONLINE
			and tdcloud.IS_AUTHENTICATED
		)

	def execute(self, context):
		node = context.node

		# Try Loading Simulation Data
		# node.sim_data = bl_cache.Signal.InvalidateCache
		sim_data = node.sim_data
		if sim_data is None:
			self.report(
				{'ERROR'},
				'Sim Data could not be loaded. Check your network connection.',
			)
		else:
			self.report({'INFO'}, 'Sim Data loaded.')

		return {'FINISHED'}


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
	output_sockets: typ.ClassVar = {
		'Sim Data': sockets.MaxwellFDTDSimDataSocketDef(),
	}

	####################
	# - Properties
	####################
	sim_data_loaded: bool = bl_cache.BLField(False)

	####################
	# - Computed
	####################
	@property
	def sim_data(self) -> td.SimulationData | None:
		cloud_task = self._compute_input(
			'Cloud Task', kind=ct.FlowKind.Value, optional=True
		)
		has_cloud_task = not ct.FlowSignal.check(cloud_task)
		if (
			has_cloud_task
			and cloud_task is not None
			and isinstance(cloud_task, tdcloud.CloudTask)
			and cloud_task.status == 'success'
		):
			sim_data = tdcloud.TidyCloudTasks.download_task_sim_data(
				cloud_task, _sim_data_cache_path(cloud_task.task_id)
			)
			self.sim_data_loaded = True
			return sim_data

		return None

	####################
	# - UI
	####################
	def draw_operators(self, _: bpy.types.Context, layout: bpy.types.UILayout):
		if self.sim_data_loaded:
			layout.operator(ct.OperatorType.NodeLoadCloudSim, text='Reload Sim')
		else:
			layout.operator(ct.OperatorType.NodeLoadCloudSim, text='Load Sim')

	####################
	# - Events
	####################
	@events.on_value_changed(
		prop_name='sim_data_loaded', run_on_init=True, props={'sim_data_loaded'}
	)
	def on_cloud_task_changed(self, props: dict):
		if props['sim_data_loaded']:
			if not self.loose_output_sockets:
				self.loose_output_sockets = {
					'Sim Data': sockets.MaxwellFDTDSimDataSocketDef(),
				}
		elif self.loose_output_sockets:
			self.loose_output_sockets = {}

	####################
	# - Output
	####################
	@events.computes_output_socket(
		'Sim Data',
		props={'sim_data_loaded'},
		input_sockets={'Cloud Task'},
	)
	def compute_sim_data(self, props: dict, input_sockets: dict) -> str:
		if props['sim_data_loaded']:
			cloud_task = input_sockets['Cloud Task']
			if (
				# Check Flow
				not ct.FlowSignal.check(cloud_task)
				# Check Task
				and cloud_task is not None
				and isinstance(cloud_task, tdcloud.CloudTask)
				and cloud_task.status == 'success'
			):
				return self.sim_data

			return ct.FlowSignal.FlowPending

		return ct.FlowSignal.FlowPending


####################
# - Blender Registration
####################
BL_REGISTER = [
	LoadCloudSim,
	Tidy3DWebImporterNode,
]
BL_NODES = {
	ct.NodeType.Tidy3DWebImporter: (ct.NodeCategory.MAXWELLSIM_INPUTS_WEBIMPORTERS)
}
