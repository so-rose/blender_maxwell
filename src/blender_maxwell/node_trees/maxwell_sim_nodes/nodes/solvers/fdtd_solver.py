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

"""Implements `FDTDSolverNode`."""

import typing as typ

import bpy

from blender_maxwell.services import tdcloud
from blender_maxwell.utils import bl_cache, logger
from blender_maxwell.utils import sympy_extra as spux

from ... import contracts as ct
from ... import sockets
from .. import base, events

log = logger.get(__name__)

FK = ct.FlowKind
FS = ct.FlowSignal
MT = spux.MathType


####################
# - Operators
####################
class RunSimulation(bpy.types.Operator):
	"""Run a Tidy3D simulation given to a `FDTDSolverNode`."""

	bl_idname = ct.OperatorType.NodeRunSimulation
	bl_label = 'Run Sim'
	bl_description = 'Run the currently tracked simulation task'

	@classmethod
	def poll(cls, context):
		"""Allow running when there are runnable tasks."""
		return (
			# Check Tidy3D Cloud
			tdcloud.IS_AUTHENTICATED
			# Check FDTDSolverNode is Accessible
			and hasattr(context, 'node')
			and hasattr(context.node, 'node_type')
			and context.node.node_type == ct.NodeType.FDTDSolver
			# Check Task is Runnable
			and context.node.are_tasks_runnable
		)

	def execute(self, context):
		"""Run all uploaded, runnable tasks."""
		node = context.node

		for cloud_task in node.cloud_tasks:
			log.debug('Submitting Cloud Task %s', cloud_task.task_id)
			cloud_task.submit()

		return {'FINISHED'}


class ReloadTrackedTask(bpy.types.Operator):
	"""Reload information of the selected task in a `FDTDSolverNode`."""

	bl_idname = ct.OperatorType.NodeReloadTrackedTask
	bl_label = 'Reload Tracked Tidy3D Cloud Task'
	bl_description = 'Reload the currently tracked simulation task'

	@classmethod
	def poll(cls, context):
		"""Always allow reloading tasks."""
		return (
			# Check Tidy3D Cloud
			tdcloud.IS_AUTHENTICATED
			# Check FDTDSolverNode is Accessible
			and hasattr(context, 'node')
			and hasattr(context.node, 'node_type')
			and context.node.node_type == ct.NodeType.FDTDSolver
		)

	def execute(self, context):
		"""Reload all tasks in all folders for which there are uploaded tasks in the node."""
		node = context.node
		for folder_id in {cloud_task.folder_id for cloud_task in node.cloud_tasks}:
			tdcloud.TidyCloudTasks.update_tasks(folder_id)

		return {'FINISHED'}


####################
# - Node
####################
class FDTDSolverNode(base.MaxwellSimNode):
	"""Solve an FDTD simulation problem using the Tidy3D cloud solver."""

	node_type = ct.NodeType.FDTDSolver
	bl_label = 'FDTD Solver'

	input_socket_sets: typ.ClassVar = {
		'Single': {
			'Cloud Task': sockets.Tidy3DCloudTaskSocketDef(
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
	output_socket_sets: typ.ClassVar = {
		'Single': {
			'Cloud Task': sockets.Tidy3DCloudTaskSocketDef(
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
	# - Properties: Incoming InfoFlow
	####################
	@events.on_value_changed(
		# Trigger
		socket_name={'Cloud Task': FK.Value, 'Cloud Tasks': FK.Array},
	)
	def on_cloud_tasks_changed(self) -> None:  # noqa: D102
		self.cloud_tasks = bl_cache.Signal.InvalidateCache

	@bl_cache.cached_bl_property()
	def cloud_tasks(self) -> list[tdcloud.CloudTask] | None:
		"""Retrieve the current cloud tasks from the input.

		If one can't be loaded, return None.
		"""
		if self.active_socket_set == 'Single':
			cloud_task_single = self._compute_input(
				'Cloud Task',
				kind=FK.Value,
			)
			has_cloud_task_single = not FS.check(cloud_task_single)
			if has_cloud_task_single and cloud_task_single is not None:
				return [cloud_task_single]

		if self.active_socket_set == 'Batch':
			cloud_task_array = self._compute_input(
				'Cloud Tasks',
				kind=FK.Array,
			)
			has_cloud_task_array = not FS.check(cloud_task_array)
			if has_cloud_task_array and all(
				cloud_task is not None for cloud_task in cloud_task_array
			):
				return cloud_task_array

		return None

	@bl_cache.cached_bl_property(depends_on={'cloud_tasks'})
	def task_infos(self) -> list[tdcloud.CloudTaskInfo | None] | None:
		"""Retrieve the current cloud task information from the input socket.

		If it can't be loaded, return None.
		"""
		if self.cloud_tasks is not None:
			task_infos = [
				tdcloud.TidyCloudTasks.task_info(cloud_task.task_id)
				for cloud_task in self.cloud_tasks
			]
			if task_infos:
				return task_infos
		return None

	@bl_cache.cached_bl_property(depends_on={'cloud_tasks'})
	def task_progress(self) -> tuple[float | None, float | None] | None:
		"""Retrieve current progress percent (in terms of time steps) and current field decay (normalized to max value).

		Either entry can be None, denoting that they aren't yet available.
		"""
		if self.cloud_tasks is not None:
			task_progress = [
				cloud_task.get_running_info() for cloud_task in self.cloud_tasks
			]
			if task_progress:
				return task_progress
		return None

	@bl_cache.cached_bl_property(depends_on={'task_progress'})
	def total_progress_pct(self) -> float | None:
		"""Retrieve current progress percent, averaged across all running tasks."""
		if self.task_progress is not None and all(
			progress[0] is not None and progress[1] is not None
			for progress in self.task_progress
		):
			return sum([progress[0] for progress in self.task_progress]) / len(
				self.task_progress
			)
		return None

	@bl_cache.cached_bl_property(depends_on={'task_infos'})
	def are_tasks_runnable(self) -> bool:
		"""Checks whether all conditions are satisfied to be able to actually run a simulation."""
		return self.task_infos is not None and all(
			task_info is not None and task_info.status == 'draft'
			for task_info in self.task_infos
		)

	####################
	# - UI
	####################
	def draw_operators(self, _, layout):
		"""Draw the button that runs the active simulation(s)."""
		# Row: Run Sim Buttons
		row = layout.row(align=True)
		row.operator(
			ct.OperatorType.NodeRunSimulation,
			text='Run Sim',
		)

	def draw_info(self, _, layout):
		"""Draw information about the running simulation."""
		tdcloud.draw_cloud_status(layout)

		# Cloud Task Info
		if self.task_infos is not None and self.task_progress is not None:
			for task_info, task_progress in zip(
				self.task_infos, self.task_progress, strict=True
			):
				if self.task_infos is not None:
					# Header
					row = layout.row()
					row.alignment = 'CENTER'
					row.label(text='Task Info')

					# Task Run Progress
					row = layout.row(align=True)
					progress_pct = (
						task_progress[0]
						if task_progress is not None and task_progress[0] is not None
						else 0.0
					)
					row.progress(
						factor=progress_pct,
						type='BAR',
						text=f'{task_info.status.capitalize()}',
					)
					row.operator(
						ct.OperatorType.NodeReloadTrackedTask,
						text='',
						icon='FILE_REFRESH',
					)

					# Task Information
					box = layout.box()

					split = box.split(factor=0.4)

					col = split.column(align=False)
					col.label(text='Status')

					col = split.column(align=False)
					col.alignment = 'RIGHT'
					col.label(text=task_info.status.capitalize())

	####################
	# - FlowKind.Value
	####################
	@events.computes_output_socket(
		'Cloud Task',
		kind=FK.Value,
		# Loaded
		props={'cloud_tasks', 'task_infos'},
	)
	def compute_cloud_task(self, props) -> str:
		"""A single simulation data object, when there only is one."""
		cloud_tasks = props['cloud_tasks']
		task_infos = props['task_infos']
		if (
			cloud_tasks is not None
			and len(cloud_tasks) == 1
			and task_infos is not None
			and task_infos[0].status == 'success'
		):
			return cloud_tasks[0]
		return FS.FlowPending

	####################
	# - FlowKind.Array
	####################
	@events.computes_output_socket(
		'Sim Datas',
		kind=FK.Array,
		# Loaded
		props={'cloud_tasks', 'task_infos'},
	)
	def compute_cloud_tasks(self, props) -> list[tdcloud.CloudTask]:
		"""All simulation data objects, for when there are more than one.

		Generally part of the same batch.
		"""
		cloud_tasks = props['cloud_tasks']
		task_infos = props['task_infos']
		if (
			cloud_tasks is not None
			and len(cloud_tasks) > 1
			and task_infos is not None
			and all(task_info.status == 'success' for task_info in task_infos)
		):
			return cloud_tasks
		return FS.FlowPending


####################
# - Blender Registration
####################
BL_REGISTER = [
	RunSimulation,
	ReloadTrackedTask,
	FDTDSolverNode,
]
BL_NODES = {ct.NodeType.FDTDSolver: (ct.NodeCategory.MAXWELLSIM_SOLVERS)}
