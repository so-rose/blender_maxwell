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

"""Implements `Tidy3DWebImporterNode`."""

import typing as typ

import bpy
import tidy3d as td

from blender_maxwell.services import tdcloud
from blender_maxwell.utils import bl_cache, logger, sim_symbols
from blender_maxwell.utils import sympy_extra as spux

from .... import contracts as ct
from .... import sockets
from ... import base, events

log = logger.get(__name__)

FK = ct.FlowKind
FS = ct.FlowSignal
MT = spux.MathType

SimDataArray: typ.TypeAlias = dict[
	tuple[sim_symbols.SimSymbol, ...], tuple[typ.Any, ...], td.SimulationData
]
SimDataArrayInfo: typ.TypeAlias = dict[
	tuple[sim_symbols.SimSymbol, ...], tuple[typ.Any, ...], typ.Any
]


####################
# - Node
####################
class Tidy3DWebImporterNode(base.MaxwellSimNode):
	"""Retrieve a simulation w/data from the Tidy3D cloud service."""

	node_type = ct.NodeType.Tidy3DWebImporter
	bl_label = 'Tidy3D Web Importer'

	input_sockets: typ.ClassVar = {
		'Preview Sim': sockets.MaxwellFDTDSimSocketDef(),
	}
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
			'Sim Data': sockets.MaxwellFDTDSimDataSocketDef(),
		},
		'Batch': {
			'Sim Datas': sockets.MaxwellFDTDSimDataSocketDef(
				active_kind=FK.Array,
			),
		},
	}

	####################
	# - Properties: Cloud Tasks -> Sim Datas
	####################
	@events.on_value_changed(
		# Trigger
		socket_name={'Cloud Task': FK.Value, 'Cloud Tasks': FK.Array},
	)
	def on_cloud_tasks_changed(self) -> None:  # noqa: D102
		self.cloud_tasks = bl_cache.Signal.InvalidateCache

	@bl_cache.cached_bl_property(depends_on={'active_socket_set'})
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
			if has_cloud_task_single:
				return [cloud_task_single]

		if self.active_socket_set == 'Batch':
			cloud_task_array = self._compute_input(
				'Cloud Tasks',
				kind=FK.Array,
			)
			has_cloud_task_array = not FS.check(cloud_task_array)
			if has_cloud_task_array:
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

	@bl_cache.cached_bl_property(depends_on={'cloud_tasks', 'task_infos'})
	def sim_datas(self) -> SimDataArray | None:
		"""Retrieve the simulation data of the current cloud task from the input socket.

		If it can't be loaded, return None.
		"""
		cloud_tasks = self.cloud_tasks
		task_infos = self.task_infos
		if (
			cloud_tasks is not None
			and task_infos is not None
			and all(task_info is not None for task_info in task_infos)
		):
			sim_datas = {}
			for cloud_task, task_info in [
				(cloud_task, task_info)
				for cloud_task, task_info in zip(cloud_tasks, task_infos, strict=True)
				if task_info.status == 'success'
			]:
				sim_data = tdcloud.TidyCloudTasks.download_task_sim_data(
					cloud_task,
					task_info.disk_cache_path(ct.addon.prefs().addon_cache_path),
				)
				if sim_data is not None:
					sim_metadata = ct.SimMetadata.from_sim(sim_data)
					sim_datas |= {sim_metadata.syms_vals: sim_data}

			if sim_datas:
				return sim_datas
		return None

	####################
	# - UI
	####################
	def draw_info(self, _: bpy.types.Context, layout: bpy.types.UILayout):
		"""Draw information about the cloud connection."""
		tdcloud.draw_cloud_status(layout)

	####################
	# - FlowKind.Value
	####################
	@events.computes_output_socket(
		'Sim Data',
		kind=FK.Value,
		# Loaded
		props={'sim_datas'},
	)
	def compute_sim_data(self, props) -> td.SimulationData | FS:
		"""A single simulation data object, when there only is one."""
		sim_datas = props['sim_datas']

		if sim_datas is not None and len(sim_datas) == 1:
			return next(iter(sim_datas.values()))
		return FS.FlowPending

	####################
	# - FlowKind.Array
	####################
	@events.computes_output_socket(
		'Sim Datas',
		kind=FK.Array,
		# Loaded
		props={'sim_datas'},
	)
	def compute_sim_datas(self, props) -> SimDataArray | FS:
		"""All simulation data objects, for when there are more than one.

		Generally part of the same batch.
		"""
		sim_datas = props['sim_datas']

		if sim_datas is not None and len(sim_datas) > 1:
			return sim_datas
		return FS.FlowPending


####################
# - Blender Registration
####################
BL_REGISTER = [
	Tidy3DWebImporterNode,
]
BL_NODES = {
	ct.NodeType.Tidy3DWebImporter: (ct.NodeCategory.MAXWELLSIM_INPUTS_WEBIMPORTERS)
}
