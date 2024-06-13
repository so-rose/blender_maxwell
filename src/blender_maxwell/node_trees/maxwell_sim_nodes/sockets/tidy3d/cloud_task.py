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

"""Implements `Tidy3DCloudTaskBLSocket`."""

import enum

import bpy

from blender_maxwell.services import tdcloud
from blender_maxwell.utils import bl_cache

from ... import contracts as ct
from .. import base


####################
# - Operators
####################
class ReloadFolderList(bpy.types.Operator):
	"""Reload the list of available folders."""

	bl_idname = ct.OperatorType.SocketReloadCloudFolderList
	bl_label = 'Reload Tidy3D Folder List'
	bl_description = 'Reload the the cached Tidy3D folder list'

	@classmethod
	def poll(cls, context):
		"""Allow reloading the folder list when online, authenticated, and attached to a `Tidy3DCloudTask` socket."""
		return (
			tdcloud.IS_ONLINE
			and tdcloud.IS_AUTHENTICATED
			and hasattr(context, 'socket')
			and hasattr(context.socket, 'socket_type')
			and context.socket.socket_type is ct.SocketType.Tidy3DCloudTask
		)

	def execute(self, context):
		"""Update the folder list, as well as any tasks attached to any existing folder ID."""
		bl_socket = context.socket

		tdcloud.TidyCloudFolders.update_folders()
		bl_socket.existing_folder_id = bl_cache.Signal.ResetEnumItems
		bl_socket.existing_folder_id = bl_cache.Signal.InvalidateCache

		if bl_socket.existing_folder_id is not None:
			tdcloud.TidyCloudTasks.update_tasks(bl_socket.existing_folder_id)

		bl_socket.existing_task_id = bl_cache.Signal.ResetEnumItems
		bl_socket.existing_task_id = bl_cache.Signal.InvalidateCache
		return {'FINISHED'}


class Authenticate(bpy.types.Operator):
	"""Authenticate with the Tidy3D API."""

	bl_idname = ct.OperatorType.SocketCloudAuthenticate
	bl_label = 'Authenticate Tidy3D'
	bl_description = 'Authenticate the Tidy3D Web API from a Cloud Task socket'

	@classmethod
	def poll(cls, context):
		"""Allow authenticating when online, not authenticated, and attached to a `Tidy3DCloudTask` socket."""
		return (
			tdcloud.IS_ONLINE
			and not tdcloud.IS_AUTHENTICATED
			and hasattr(context, 'socket')
			and hasattr(context.socket, 'socket_type')
			and context.socket.socket_type is ct.SocketType.Tidy3DCloudTask
		)

	def execute(self, context):
		"""Try authenticating the socket with the web service."""
		bl_socket = context.socket

		if not tdcloud.check_authentication():
			tdcloud.authenticate_with_api_key(bl_socket.api_key)
			bl_socket.api_key = ''

			bl_socket.existing_folder_id = bl_cache.Signal.ResetEnumItems
			bl_socket.existing_folder_id = bl_cache.Signal.InvalidateCache
			bl_socket.existing_task_id = bl_cache.Signal.ResetEnumItems
			bl_socket.existing_task_id = bl_cache.Signal.InvalidateCache

		return {'FINISHED'}


####################
# - Socket
####################
class Tidy3DCloudTaskBLSocket(base.MaxwellSimSocket):
	"""Interact with Tidy3D Cloud Tasks.

	Attributes:
		api_key: API key for the Tidy3D cloud.
		should_exist: Whether or not the cloud task should already exist.
		existing_folder_id: ID of an existing folder on the Tidy3D cloud.
		existing_task_id: ID of an existing task on the Tidy3D cloud.
		new_task_name: Name of a new task to submit to the Tidy3D cloud.
	"""

	socket_type = ct.SocketType.Tidy3DCloudTask
	bl_label = 'Tidy3D Cloud Task'

	####################
	# - Properties: Authentication
	####################
	api_key: str = bl_cache.BLField('', str_secret=True)

	####################
	# - Properties: Existance
	####################
	should_exist: bool = bl_cache.BLField(False)
	new_task_name: str = bl_cache.BLField('')

	####################
	# - Properties: Cloud Folders
	####################
	existing_folder_id: enum.StrEnum = bl_cache.BLField(
		enum_cb=lambda self, _: self.search_cloud_folders()
	)

	def search_cloud_folders(self) -> list[ct.BLEnumElement]:
		"""Get all Tidy3D cloud folders."""
		if tdcloud.IS_AUTHENTICATED:
			return [
				(
					folder_id,
					cloud_folder.folder_name,
					f'Folder {cloud_folder.folder_name} (ID={folder_id})',
					'',
					i,
				)
				for i, (folder_id, cloud_folder) in enumerate(
					tdcloud.TidyCloudFolders.folders().items()
				)
			]

		return []

	####################
	# - Properties: Cloud Tasks
	####################
	existing_task_id: enum.StrEnum = bl_cache.BLField(
		enum_cb=lambda self, _: self.search_cloud_tasks(),
		cb_depends_on={'existing_folder_id'},
	)

	def search_cloud_tasks(self) -> list[ct.BLEnumElement]:
		"""Retrieve all Tidy3D cloud folders from the cloud service."""
		if self.existing_folder_id is None or not tdcloud.IS_AUTHENTICATED:
			return []

		# Get Cloud Folder
		cloud_folder = tdcloud.TidyCloudFolders.folders().get(self.existing_folder_id)
		if cloud_folder is None:
			return []

		# Get Cloud Tasks
		tasks = tdcloud.TidyCloudTasks.tasks(cloud_folder)
		return [
			(
				## Task ID
				task.task_id,
				## Task Dropdown Names
				' '.join(
					[
						task.taskName,
						'('
						+ task.created_at.astimezone().strftime('%y-%m-%d @ %H:%M %Z')
						+ ')',
					]
				),
				## Task Description
				f'Task Status: {task.status}',
				## Status Icon
				icon
				if (
					icon := {
						'draft': 'SEQUENCE_COLOR_08',
						'initialized': 'SHADING_SOLID',
						'queued': 'SEQUENCE_COLOR_03',
						'preprocessing': 'SEQUENCE_COLOR_02',
						'running': 'SEQUENCE_COLOR_05',
						'postprocessing': 'SEQUENCE_COLOR_06',
						'success': 'SEQUENCE_COLOR_04',
						'error': 'SEQUENCE_COLOR_01',
					}.get(task.status)
				)
				else 'SEQUENCE_COLOR_09',
				## Unique Number
				i,
			)
			for i, task in enumerate(
				sorted(
					tasks.values(),
					key=lambda el: el.created_at,
					reverse=True,
				)
			)
		]

	####################
	# - FlowKinds
	####################
	@bl_cache.cached_bl_property(depends_on={'active_kind', 'should_exist'})
	def capabilities(self) -> ct.CapabilitiesFlow:
		"""Cloud task sockets are compatible by presumtion of existance."""
		return ct.CapabilitiesFlow(
			socket_type=self.socket_type,
			active_kind=self.active_kind,
			must_match={'should_exist': self.should_exist},
		)

	@bl_cache.cached_bl_property(
		depends_on={
			'should_exist',
			'new_task_name',
			'existing_folder_id',
			'existing_task_id',
		}
	)
	def value(
		self,
	) -> ct.NewSimCloudTask | tdcloud.CloudTask | ct.FlowSignal:
		"""Return either the existing cloud task, or an object describing how to make it."""
		if tdcloud.IS_AUTHENTICATED:
			# Retrieve Folder
			cloud_folder = tdcloud.TidyCloudFolders.folders().get(
				self.existing_folder_id
			)
			if cloud_folder is not None:
				# Case: New Task
				if not self.should_exist:
					return ct.NewSimCloudTask(
						task_name=self.new_task_name, cloud_folder=cloud_folder
					)

				# Case: Existing Task
				if self.existing_task_id is not None:
					cloud_task = tdcloud.TidyCloudTasks.tasks(cloud_folder)[
						self.existing_task_id
					]
					if cloud_task is not None:
						return cloud_task
			else:
				return ct.FlowSignal.NoFlow  ## Folder deleted somewhere else
		return ct.FlowSignal.FlowPending

	####################
	# - UI
	####################
	def draw_prelock(
		self,
		_: bpy.types.Context,
		col: bpy.types.UILayout,
		node: bpy.types.Node,  # noqa: ARG002
		text: str,  # noqa: ARG002
	) -> None:
		"""Draw a lock-immune interface for authenticating with the Tidy3D API, when the authentication is invalid."""
		if not tdcloud.IS_AUTHENTICATED:
			row = col.row()
			row.alignment = 'CENTER'
			row.label(text='Tidy3D API Key')

			row = col.row()
			row.prop(self, self.blfields['api_key'], text='')

			row = col.row()
			row.operator(
				ct.OperatorType.SocketCloudAuthenticate,
				text='Connect',
			)

	def draw_value(self, col: bpy.types.UILayout) -> None:
		"""When authenticated, draw the node UI."""
		if not tdcloud.IS_AUTHENTICATED:
			return

		# Cloud Folder Selector
		row = col.row()
		row.label(icon='FILE_FOLDER')
		row.prop(self, self.blfields['existing_folder_id'], text='')
		row.operator(
			ct.OperatorType.SocketReloadCloudFolderList,
			text='',
			icon='FILE_REFRESH',
		)

		# New Task Name Selector
		row = col.row()
		if not self.should_exist:
			row = col.row()
			row.label(icon='NETWORK_DRIVE')
			row.prop(self, self.blfields['new_task_name'], text='')

			col.separator(factor=1.0)

			box = col.box()
			row = box.row()

		row.prop(self, self.blfields['existing_task_id'], text='')


####################
# - Socket Configuration
####################
class Tidy3DCloudTaskSocketDef(base.SocketDef):
	"""Declarative object guiding the creation of a `Tidy3DCloudTask` socket."""

	socket_type: ct.SocketType = ct.SocketType.Tidy3DCloudTask

	should_exist: bool

	def init(self, bl_socket: Tidy3DCloudTaskBLSocket) -> None:
		"""Initialize the passed socket with the properties of this object."""
		bl_socket.should_exist = self.should_exist
		bl_socket.use_prelock = True


####################
# - Blender Registration
####################
BL_REGISTER = [
	ReloadFolderList,
	Authenticate,
	Tidy3DCloudTaskBLSocket,
]
