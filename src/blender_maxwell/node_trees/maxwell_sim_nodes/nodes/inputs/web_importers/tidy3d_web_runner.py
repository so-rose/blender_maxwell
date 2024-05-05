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

import typing as typ

import bpy
import tidy3d as td

from blender_maxwell.services import tdcloud
from blender_maxwell.utils import bl_cache, logger

from .... import contracts as ct
from .... import sockets
from ... import base, events

log = logger.get(__name__)


####################
# - Operators
####################
class RunSimulation(bpy.types.Operator):
	"""Run a Tidy3D simulation accessible from a `Tidy3DWebRunnerNode`."""

	bl_idname = ct.OperatorType.NodeRunSimulation
	bl_label = 'Run Sim'
	bl_description = 'Run the currently tracked simulation task'

	@classmethod
	def poll(cls, context):
		return (
			# Check Tidy3D Cloud
			tdcloud.IS_AUTHENTICATED
			# Check Tidy3DWebRunnerNode is Accessible
			and hasattr(context, 'node')
			and hasattr(context.node, 'node_type')
			and context.node.node_type == ct.NodeType.Tidy3DWebRunner
			# Check Task is Runnable
			and context.node.is_task_runnable
		)

	def execute(self, context):
		node = context.node
		node.cloud_task.submit()

		return {'FINISHED'}


class ReloadTrackedTask(bpy.types.Operator):
	"""Reload information of the selected task in a `Tidy3DWebRunnerNode`."""

	bl_idname = ct.OperatorType.NodeReloadTrackedTask
	bl_label = 'Reload Tracked Tidy3D Cloud Task'
	bl_description = 'Reload the currently tracked simulation task'

	@classmethod
	def poll(cls, context):
		return (
			# Check Tidy3D Cloud
			tdcloud.IS_AUTHENTICATED
			# Check Tidy3DWebRunnerNode is Accessible
			and hasattr(context, 'node')
			and hasattr(context.node, 'node_type')
			and context.node.node_type == ct.NodeType.Tidy3DWebRunner
		)

	def execute(self, context):
		node = context.node
		tdcloud.TidyCloudTasks.update_task(node.cloud_task)
		node.sim_data = bl_cache.Signal.InvalidateCache

		return {'FINISHED'}


####################
# - Node
####################
class Tidy3DWebRunnerNode(base.MaxwellSimNode):
	node_type = ct.NodeType.Tidy3DWebRunner
	bl_label = 'Tidy3D Web Runner'

	input_sockets: typ.ClassVar = {
		'Cloud Task': sockets.Tidy3DCloudTaskSocketDef(
			should_exist=True,  ## Ensure it is never NewSimCloudTask
		),
	}
	output_sockets: typ.ClassVar = {
		'Sim Data': sockets.MaxwellFDTDSimDataSocketDef(),
	}

	####################
	# - Computed (Cached)
	####################
	@property
	def cloud_task(self) -> tdcloud.CloudTask | None:
		"""Retrieve the current cloud task from the input socket.

		If one can't be loaded, return None.
		"""
		cloud_task = self._compute_input(
			'Cloud Task',
			kind=ct.FlowKind.Value,
		)
		has_cloud_task = not ct.FlowSignal.check(cloud_task)

		if has_cloud_task:
			return cloud_task
		return None

	@property
	def task_info(self) -> tdcloud.CloudTaskInfo | None:
		"""Retrieve the current cloud task information from the input socket.

		If it can't be loaded, return None.
		"""
		cloud_task = self.cloud_task
		if cloud_task is None:
			return None

		# Retrieve Task Info
		task_info = tdcloud.TidyCloudTasks.task_info(cloud_task.task_id)
		if task_info is None:
			return None

		return task_info

	@bl_cache.cached_bl_property(persist=False)
	def sim_data(self) -> td.Simulation | None:
		"""Retrieve the simulation data of the current cloud task from the input socket.

		If it can't be loaded, return None.
		"""
		task_info = self.task_info
		if task_info is None:
			return None

		if task_info.status == 'success':
			# Download Sim Data
			## -> self.cloud_task really shouldn't be able to be None here.
			## -> So, we check it by applying the Ostrich method.
			sim_data = tdcloud.TidyCloudTasks.download_task_sim_data(
				self.cloud_task,
				tdcloud.TidyCloudTasks.task_info(
					self.cloud_task.task_id
				).disk_cache_path(ct.addon.ADDON_CACHE),
			)
			if sim_data is None:
				return None

			return sim_data

		return None

	####################
	# - Computed (Uncached)
	####################
	@property
	def is_task_runnable(self) -> bool:
		"""Checks whether all conditions are satisfied to be able to actually run a simulation."""
		if self.task_info is not None:
			return self.task_info.status == 'draft'
		return False

	####################
	# - UI
	####################
	def draw_operators(self, context, layout):
		# Row: Run Sim Buttons
		row = layout.row(align=True)
		row.operator(
			ct.OperatorType.NodeRunSimulation,
			text='Run Sim',
		)

	def draw_info(self, context, layout):
		# Connection Info
		auth_icon = 'CHECKBOX_HLT' if tdcloud.IS_AUTHENTICATED else 'CHECKBOX_DEHLT'
		conn_icon = 'CHECKBOX_HLT' if tdcloud.IS_ONLINE else 'CHECKBOX_DEHLT'

		row = layout.row()
		row.alignment = 'CENTER'
		row.label(text='Cloud Status')
		box = layout.box()
		split = box.split(factor=0.85)

		## Split: Left Column
		col = split.column(align=False)
		col.label(text='Authed')
		col.label(text='Connected')

		## Split: Right Column
		col = split.column(align=False)
		col.label(icon=auth_icon)
		col.label(icon=conn_icon)

		# Cloud Task Info
		if self.task_info is not None:
			# Header
			row = layout.row()
			row.alignment = 'CENTER'
			row.label(text='Task Info')

			# Task Run Progress
			# row = layout.row(align=True)
			# row.progress(
			# factor=0.0,
			# type='BAR',
			# text=f'Status: {self.task_info.status.capitalize()}',
			# )
			row.operator(
				ct.OperatorType.NodeReloadTrackedTask,
				text='',
				icon='FILE_REFRESH',
			)

			# Task Information
			box = layout.box()
			split = box.split(factor=0.4)

			## Split: Left Column
			col = split.column(align=False)
			col.label(text='Status')
			col.label(text='Real Cost')

			## Split: Right Column
			cost_real = (
				f'{self.task_info.cost_real:.2f}'
				if self.task_info.cost_real is not None
				else 'TBD'
			)

			col = split.column(align=False)
			col.alignment = 'RIGHT'
			col.label(text=self.task_info.status.capitalize())
			col.label(text=f'{cost_real} creds')

	####################
	# - Output Methods
	####################
	@events.on_value_changed(
		socket_name='Cloud Task',
	)
	def compute_cloud_task(self) -> None:
		self.sim_data = bl_cache.Signal.InvalidateCache

	@events.computes_output_socket(
		'Sim Data',
		props={'sim_data'},
		input_sockets={'Cloud Task'},  ## Keep to respect dependency chains.
	)
	def compute_sim_data(
		self, props, input_sockets
	) -> td.SimulationData | ct.FlowSignal:
		if props['sim_data'] is None:
			return ct.FlowSignal.FlowPending

		return props['sim_data']


####################
# - Blender Registration
####################
BL_REGISTER = [
	RunSimulation,
	ReloadTrackedTask,
	Tidy3DWebRunnerNode,
]
BL_NODES = {
	ct.NodeType.Tidy3DWebRunner: (ct.NodeCategory.MAXWELLSIM_INPUTS_WEBIMPORTERS)
}
