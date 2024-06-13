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

"""Implements `Tidy3DWebExporter`."""

import typing as typ

import bpy
import tidy3d as td

from blender_maxwell.services import tdcloud
from blender_maxwell.utils import bl_cache, logger, sim_symbols

from .... import contracts as ct
from .... import sockets
from ... import base, events

log = logger.get(__name__)

FK = ct.FlowKind
FS = ct.FlowSignal

SimArray: typ.TypeAlias = dict[
	tuple[sim_symbols.SimSymbol, ...], tuple[typ.Any, ...], td.Simulation
]
SimArrayInfo: typ.TypeAlias = dict[
	tuple[sim_symbols.SimSymbol, ...], tuple[typ.Any, ...], td.Simulation
]


####################
# - Operators
####################
class UploadSimulation(bpy.types.Operator):
	"""Upload the simulation embedded in the `Tidy3DWebExpoerter`."""

	bl_idname = ct.OperatorType.NodeUploadSimulation
	bl_label = 'Upload Tidy3D Simulation'
	bl_description = 'Upload the attached (locked) simulation, such that it is ready to run on the Tidy3D cloud'

	@classmethod
	def poll(cls, context):
		"""Allow running whenever there are simulations to upload."""
		return (
			# Check Tidy3D Cloud
			tdcloud.IS_AUTHENTICATED
			# Check Tidy3DWebExporter is Accessible
			and hasattr(context, 'node')
			and hasattr(context.node, 'node_type')
			and context.node.node_type == ct.NodeType.Tidy3DWebExporter
			# Check Sim is Available (aka. uploadeable)
			and context.node.sims
			and context.node.sims_uploadable
			and not context.node.uploaded_task_ids
		)

	def execute(self, context):
		"""Upload either a single or a batch of simulations.

		Later retrieval of realized parameter points exist in the form of a realized serializable dictionary attached to the `.attrs` field of any `td.Simulation` object.
		"""
		node = context.node

		if node.base_cloud_task is not None:
			base_task_name = node.base_cloud_task.task_name
			base_task_folder = node.base_cloud_task.cloud_folder
		else:
			self.report({'ERROR'}, 'No base cloud task name')
			return {'FINISHED'}

		if node.active_socket_set == 'Single':
			if len(list(node.sims.values())) == 1:
				sim = next(iter(node.sims.values()))
			else:
				self.report({'ERROR'}, '>1 sims for "Single"-mode sim exporter.')
				return {'FINISHED'}

			cloud_task = tdcloud.TidyCloudTasks.mk_task(
				task_name=base_task_name,
				cloud_folder=base_task_folder,
				sim=sim,
				verbose=True,
			)
			node.uploaded_task_ids = (cloud_task.task_id,)

		if node.active_socket_set == 'Batch':
			cloud_tasks = [
				tdcloud.TidyCloudTasks.mk_task(
					task_name=base_task_name + f'_{i}',
					cloud_folder=base_task_folder,
					sim=sim,
					verbose=True,
				)
				for i, sim in enumerate(node.sims.values())
			]
			node.uploaded_task_ids = tuple(
				[cloud_task.task_id for cloud_task in cloud_tasks]
			)

		return {'FINISHED'}


class ReleaseUploadedTask(bpy.types.Operator):
	"""Release the uploaded simulation embedded in the `Tidy3DWebExpoerter`."""

	bl_idname = ct.OperatorType.NodeReleaseUploadedTask
	bl_label = 'Release Tracked Tidy3D Cloud Task'
	bl_description = 'Release the currently tracked simulation task'

	@classmethod
	def poll(cls, context):
		"""Allow running whenever a particular FDTDSim node is tracking uploaded simulations."""
		return (
			# Check Tidy3DWebExporter is Accessible
			hasattr(context, 'node')
			and hasattr(context.node, 'node_type')
			and context.node.node_type == ct.NodeType.Tidy3DWebExporter
			# Check Sim is Available (aka. uploadeable)
			and context.node.uploaded_task_ids
		)

	def execute(self, context):
		"""Invalidate the `.sims` property, triggering reevaluation of all downstream information about the simulation."""
		node = context.node
		node.uploaded_task_ids = ()
		return {'FINISHED'}


####################
# - Node
####################
class Tidy3DWebExporterNode(base.MaxwellSimNode):
	"""Export a simulation to the Tidy3D cloud service, where it can be queried and run."""

	node_type = ct.NodeType.Tidy3DWebExporter
	bl_label = 'Tidy3D Web Exporter'

	input_socket_sets: typ.ClassVar = {
		'Single': {
			'Sim': sockets.MaxwellFDTDSimSocketDef(),
			'Cloud Task': sockets.Tidy3DCloudTaskSocketDef(
				active_kind=FK.Value,
				should_exist=False,
			),
		},
		'Batch': {
			'Sims': sockets.MaxwellFDTDSimSocketDef(
				active_kind=FK.Array,
			),
			'Cloud Task': sockets.Tidy3DCloudTaskSocketDef(
				active_kind=FK.Value,
				should_exist=False,
			),
		},
	}
	output_socket_sets: typ.ClassVar = {
		'Single': {
			'Cloud Task': sockets.Tidy3DCloudTaskSocketDef(
				active_kind=FK.Value,
				should_exist=True,
			),
		},
		'Batch': {
			'Cloud Tasks': sockets.Tidy3DCloudTaskSocketDef(
				active_kind=FK.Array,
				should_exist=True,
			),
		},
	}

	####################
	# - Properties
	####################
	uploaded_task_ids: tuple[str, ...] = bl_cache.BLField(())

	####################
	# - Properties: Socket -> Props
	####################
	@events.on_value_changed(
		socket_name={'Sim': {FK.Value, FK.Array}},
	)
	def on_sims_changed(self) -> None:
		"""Regenerate the simulation property on changes."""
		self.sims = bl_cache.Signal.InvalidateCache

	@events.on_value_changed(
		socket_name={'Cloud Task': {FK.Value}},
	)
	def on_base_cloud_task_changed(self) -> None:
		"""Regenerate the cloud task property on changes."""
		self.base_cloud_task = bl_cache.Signal.InvalidateCache

	####################
	# - Properties: Socket Alias
	####################
	@bl_cache.cached_bl_property(depends_on={'active_socket_set'})
	def sims(self) -> SimArray | None:
		"""The complete description of all simulation data input."""
		if self.active_socket_set == 'Single':
			sim_data_value = self._compute_input('Sim', kind=FK.Value)
			has_sim_data_value = not FS.check(sim_data_value)
			if has_sim_data_value:
				return {(): sim_data_value}

		elif self.active_socket_set == 'Batch':
			sim_data_array = self._compute_input('Sims', kind=FK.Array)
			has_sim_data_array = not FS.check(sim_data_array)
			if has_sim_data_array:
				return sim_data_array

		return None

	@bl_cache.cached_bl_property()
	def base_cloud_task(self) -> ct.NewSimCloudTask | None:
		"""The complete description of all simulation input objects."""
		base_cloud_task = self._compute_input(
			'Cloud Task',
			kind=ct.FlowKind.Value,
		)
		has_base_cloud_task = not FS.check(base_cloud_task)
		if has_base_cloud_task and base_cloud_task.task_name != '':
			return base_cloud_task
		return None

	####################
	# - Properties: Simulations
	####################
	@bl_cache.cached_bl_property(depends_on={'sims'})
	def sims_valid(
		self,
	) -> (
		dict[tuple[tuple[sim_symbols.SimSymbol, ...], tuple[typ.Any, ...]], bool] | None
	):
		"""Whether all sims are valid."""
		if self.sims is not None:
			validity = {}
			for k, sim in self.sims.items():  # noqa: B007
				try:
					pass  ## TODO: VERY slow, batch checking is infeasible
					# sim.validate_pre_upload(source_required=True)
				except td.exceptions.SetupError:
					validity[k] = False
				else:
					validity[k] = True

			return validity
		return None

	####################
	# - Properties: Tasks
	####################
	@bl_cache.cached_bl_property(depends_on={'uploaded_task_ids'})
	def uploaded_task_infos(self) -> list[tdcloud.CloudTask | None] | None:
		"""Retrieve information about the uploaded cloud tasks."""
		if self.uploaded_task_ids:
			return [
				tdcloud.TidyCloudTasks.task_info(task_id)
				for task_id in self.uploaded_task_ids
			]
		return None

	@bl_cache.cached_bl_property(depends_on={'uploaded_task_infos'})
	def est_costs(self) -> list[float | None] | None:
		"""Estimate the FlexCredit cost of each uploaded task."""
		if self.uploaded_task_infos is not None and all(
			task_info is not None for task_info in self.uploaded_task_infos
		):
			return [task_info.cost_est() for task_info in self.uploaded_task_infos]
		return None

	@bl_cache.cached_bl_property(depends_on={'est_costs'})
	def total_est_cost(self) -> list[float | None] | None:
		"""Estimate the total FlexCredits cost of all uploaded tasks."""
		if self.est_costs is not None and all(
			est_cost is not None for est_cost in self.est_costs
		):
			return sum(self.est_costs)
		return None

	@bl_cache.cached_bl_property(depends_on={'uploaded_task_infos'})
	def real_costs(self) -> list[float | None] | None:
		"""Estimate the FlexCredit cost of each uploaded task."""
		if self.uploaded_task_infos is not None and all(
			task_info is not None for task_info in self.uploaded_task_infos
		):
			return [task_info.cost_real for task_info in self.uploaded_task_infos]
		return None

	@bl_cache.cached_bl_property(depends_on={'real_costs'})
	def total_real_cost(self) -> list[float | None] | None:
		"""Estimate the total FlexCredits cost of all uploaded tasks."""
		if self.real_costs is not None and all(
			real_cost is not None for real_cost in self.real_costs
		):
			return sum(self.real_costs)
		return None

	####################
	# - Computed - Combined
	####################
	@bl_cache.cached_bl_property(depends_on={'sims_valid'})
	def sims_uploadable(self) -> bool:
		"""Whether all simulations can be uploaded."""
		return self.sims_valid is not None and all(self.sims_valid.values())

	####################
	# - UI
	####################
	@bl_cache.cached_bl_property(
		depends_on={
			'uploaded_task_infos',
			'total_est_cost',
			'total_real_cost',
		}
	)
	def task_labels(self) -> SimArrayInfo | None:
		"""Pre-processed labels for efficient drawing of task info."""
		if self.uploaded_task_infos is not None and all(
			task_info is not None for task_info in self.uploaded_task_infos
		):
			return {
				task_info.task_id: [
					f'Task: {task_info.task_name}',
					('Status', task_info.status),
					(
						'Est.',
						(
							f'{self.total_est_cost:.2f} creds'
							if self.total_est_cost is not None
							else 'TBD...'
						),
					),
					(
						'Real',
						(
							f'{self.total_real_cost:.2f} creds'
							if self.total_real_cost is not None
							else 'TBD'
						),
					),
				]
				for task_info in self.uploaded_task_infos
			}
		return None

	def draw_operators(self, _, layout):
		"""Draw operators for uploading/releasing simulations."""
		# Row: Upload Sim Buttons
		row = layout.row(align=True)
		row.operator(
			ct.OperatorType.NodeUploadSimulation,
			text='Upload',
		)
		if self.uploaded_task_ids:
			row.operator(
				ct.OperatorType.NodeReleaseUploadedTask,
				icon='LOOP_BACK',
				text='',
			)

	def draw_info(self, _, layout):
		"""Draw information relevant for simulation uploading."""
		# Connection Info
		auth_icon = 'CHECKBOX_HLT' if tdcloud.IS_AUTHENTICATED else 'CHECKBOX_DEHLT'
		conn_icon = 'CHECKBOX_HLT' if tdcloud.IS_ONLINE else 'CHECKBOX_DEHLT'

		box = layout.box()

		# Cloud Info
		row = box.row()
		row.alignment = 'CENTER'
		row.label(text='Cloud Status')

		split = box.split(factor=0.85)

		col = split.column(align=False)
		col.label(text='Authed')
		col.label(text='Connected')

		col = split.column(align=False)
		col.label(icon=auth_icon)
		col.label(icon=conn_icon)

		if self.task_labels is not None:
			for labels in self.task_labels.values():
				row = layout.row(align=True)
				row.alignment = 'CENTER'
				row.label(text='Task Status')

				for el in labels:
					# Header
					if isinstance(el, str):
						box = layout.box()
						row = box.row(align=True)
						row.alignment = 'CENTER'
						row.label(text=el)

						split = box.split(factor=0.4)
						col_l = split.column(align=True)
						col_r = split.column(align=True)

					# Label Pair
					elif isinstance(el, tuple):
						col_l.label(text=el[0])
						col_r.label(text=el[1])

					else:
						raise TypeError

				break

	####################
	# - Events
	####################
	@events.on_value_changed(
		prop_name='uploaded_task_ids',
		# Loaded
		props={'uploaded_task_ids'},
	)
	def on_uploaded_task_changed(self, props):
		"""When uploaded tasks change, take appropriate action.

		- Enable/Disable Lock: To prevent node-tree modifications that would invalidate the validity of uploaded tasks.
		- Ensure Est Cost: Repeatedly try to load the estimated cost of all tasks, until all are available.
		"""
		uploaded_task_ids = props['uploaded_task_ids']

		# Lock
		if uploaded_task_ids:
			self.trigger_event(ct.FlowEvent.EnableLock)
			self.locked = False

			# Force Computation of Estimated Cost
			## -> Try recomputing the estimated cost of all tasks.
			## -> Once all are non-None, stop.
			max_tries = 20
			for _ in range(max_tries):
				self.est_costs = bl_cache.Signal.InvalidateCache
				if self.total_est_cost is not None:
					break

		else:
			self.trigger_event(ct.FlowEvent.DisableLock)

	####################
	# - FlowKind.Value
	####################
	@events.computes_output_socket(
		'Cloud Task',
		kind=ct.FlowKind.Value,
		# Loaded
		props={'uploaded_task_ids'},
	)
	def compute_cloud_task(self, props) -> tdcloud.CloudTask | None:
		"""A single uploaded cloud task, when there only is one."""
		uploaded_task_ids = props['uploaded_task_ids']

		if uploaded_task_ids is not None and len(uploaded_task_ids) == 1:
			cloud_task = tdcloud.TidyCloudTasks.task(uploaded_task_ids[0])
			if cloud_task is not None:
				return cloud_task
		return FS.FlowPending

	####################
	# - FlowKind.Array
	####################
	@events.computes_output_socket(
		'Cloud Tasks',
		kind=ct.FlowKind.Array,
		# Loaded
		props={'uploaded_task_ids'},
	)
	def compute_cloud_tasks(self, props) -> tdcloud.CloudTask | None:
		"""All uploaded cloud task, when there are more than one."""
		uploaded_task_ids = props['uploaded_task_ids']

		if len(uploaded_task_ids) > 1:
			cloud_tasks = [
				tdcloud.TidyCloudTasks.task(task_id) for task_id in uploaded_task_ids
			]
			if all(cloud_task is not None for cloud_task in cloud_tasks):
				return cloud_tasks
		return FS.FlowPending


####################
# - Blender Registration
####################
BL_REGISTER = [
	UploadSimulation,
	ReleaseUploadedTask,
	Tidy3DWebExporterNode,
]
BL_NODES = {
	ct.NodeType.Tidy3DWebExporter: (ct.NodeCategory.MAXWELLSIM_OUTPUTS_WEBEXPORTERS)
}
