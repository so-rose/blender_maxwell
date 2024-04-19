import bpy

from .....services import tdcloud
from ... import contracts as ct
from .. import base


####################
# - Operators
####################
class ReloadFolderList(bpy.types.Operator):
	bl_idname = 'blender_maxwell.sockets__reload_folder_list'
	bl_label = 'Reload Tidy3D Folder List'
	bl_description = 'Reload the the cached Tidy3D folder list'

	@classmethod
	def poll(cls, context):
		return (
			tdcloud.IS_AUTHENTICATED
			and hasattr(context, 'socket')
			and hasattr(context.socket, 'socket_type')
			and context.socket.socket_type == ct.SocketType.Tidy3DCloudTask
		)

	def execute(self, context):
		socket = context.socket

		tdcloud.TidyCloudFolders.update_folders()
		tdcloud.TidyCloudTasks.update_tasks(socket.existing_folder_id)

		return {'FINISHED'}


class Authenticate(bpy.types.Operator):
	bl_idname = 'blender_maxwell.sockets__authenticate'
	bl_label = 'Authenticate Tidy3D'
	bl_description = 'Authenticate the Tidy3D Web API from a Cloud Task socket'

	@classmethod
	def poll(cls, context):
		return (
			not tdcloud.IS_AUTHENTICATED
			and hasattr(context, 'socket')
			and hasattr(context.socket, 'socket_type')
			and context.socket.socket_type == ct.SocketType.Tidy3DCloudTask
		)

	def execute(self, context):
		bl_socket = context.socket

		if not tdcloud.check_authentication():
			tdcloud.authenticate_with_api_key(bl_socket.api_key)
			bl_socket.api_key = ''

		return {'FINISHED'}


####################
# - Socket
####################
class Tidy3DCloudTaskBLSocket(base.MaxwellSimSocket):
	socket_type = ct.SocketType.Tidy3DCloudTask
	bl_label = 'Tidy3D Cloud Task'

	use_prelock = True

	####################
	# - Properties
	####################
	# Authentication
	api_key: bpy.props.StringProperty(
		name='API Key',
		description='API Key for the Tidy3D Cloud',
		default='',
		options={'SKIP_SAVE'},
		subtype='PASSWORD',
	)

	# Task Existance Presumption
	should_exist: bpy.props.BoolProperty(
		name='Cloud Task Should Exist',
		description='Whether or not the cloud task should already exist',
		default=False,
	)

	# Identifiers
	existing_folder_id: bpy.props.EnumProperty(
		name='Folder of Cloud Tasks',
		description='An existing folder on the Tidy3D Cloud',
		items=lambda self, _: self.retrieve_folders(),
		update=(lambda self, context: self.on_prop_changed('existing_folder_id', context)),
	)
	existing_task_id: bpy.props.EnumProperty(
		name='Existing Cloud Task',
		description='An existing task on the Tidy3D Cloud, within the given folder',
		items=lambda self, _: self.retrieve_tasks(),
		update=(lambda self, context: self.on_prop_changed('existing_task_id', context)),
	)

	# (Potential) New Task
	new_task_name: bpy.props.StringProperty(
		name='New Cloud Task Name',
		description='Name of a new task to submit to the Tidy3D Cloud',
		default='',
		update=(lambda self, context: self.on_prop_changed('new_task_name', context)),
	)

	####################
	# - Property Methods
	####################
	def sync_existing_folder_id(self, context):
		folder_task_ids = self.retrieve_tasks()

		self.existing_task_id = folder_task_ids[0][0]
		## There's guaranteed to at least be one element, even if it's "NONE".

		self.on_prop_changed('existing_folder_id', context)

	def retrieve_folders(self) -> list[tuple]:
		folders = tdcloud.TidyCloudFolders.folders()
		if not folders:
			return [('NONE', 'None', 'No folders')]

		return [
			(
				cloud_folder.folder_id,
				cloud_folder.folder_name,
				f"Folder 'cloud_folder.folder_name' with ID {folder_id}",
			)
			for folder_id, cloud_folder in folders.items()
		]

	def retrieve_tasks(self) -> list[tuple]:
		if (
			cloud_folder := tdcloud.TidyCloudFolders.folders().get(
				self.existing_folder_id
			)
		) is None:
			return [('NONE', 'None', "Folder doesn't exist")]

		tasks = tdcloud.TidyCloudTasks.tasks(cloud_folder)
		if not tasks:
			return [('NONE', 'None', 'No tasks in folder')]

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
				_icon
				if (
					_icon := {
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
	# - Task Sync Methods
	####################
	def sync_created_new_task(self, cloud_task):
		"""Called whenever the task specified in `new_task_name` has been actually created.

		This changes the socket somewhat: Folder/task IDs are set, and the socket is switched to presume that the task exists.

		If the socket is linked, then an error is raised.
		"""
		# Propagate along Link
		if self.is_linked:
			msg = 'Cannot sync newly created task to linked Cloud Task socket.'
			raise ValueError(msg)
			## TODO: A little aggressive. Is there a good use case?

		# Synchronize w/New Task Information
		self.existing_folder_id = cloud_task.folder_id
		self.existing_task_id = cloud_task.task_id
		self.should_exist = True

	def sync_prepare_new_task(self):
		"""Called to switch the socket to no longer presume that the task it specifies exists (yet).

		If the socket is linked, then an error is raised.
		"""
		# Propagate along Link
		if self.is_linked:
			msg = 'Cannot sync newly created task to linked Cloud Task socket.'
			raise ValueError(msg)
			## TODO: A little aggressive. Is there a good use case?

		# Synchronize w/New Task Information
		self.should_exist = False

	####################
	# - Socket UI
	####################
	def draw_label_row(self, row: bpy.types.UILayout, text: str):
		row.label(text=text)

		auth_icon = 'LOCKVIEW_ON' if tdcloud.IS_AUTHENTICATED else 'LOCKVIEW_OFF'
		row.operator(
			Authenticate.bl_idname,
			text='',
			icon=auth_icon,
		)

	def draw_prelock(
		self,
		context: bpy.types.Context,
		col: bpy.types.UILayout,
		node: bpy.types.Node,
		text: str,
	) -> None:
		if not tdcloud.IS_AUTHENTICATED:
			row = col.row()
			row.alignment = 'CENTER'
			row.label(text='Tidy3D API Key')

			row = col.row()
			row.prop(self, 'api_key', text='')

			row = col.row()
			row.operator(
				Authenticate.bl_idname,
				text='Connect',
			)

	def draw_value(self, col: bpy.types.UILayout) -> None:
		if not tdcloud.IS_AUTHENTICATED:
			return

		# Cloud Folder Selector
		row = col.row()
		row.label(icon='FILE_FOLDER')
		row.prop(self, 'existing_folder_id', text='')
		row.operator(
			ReloadFolderList.bl_idname,
			text='',
			icon='FILE_REFRESH',
		)

		# New Task Name Selector
		row = col.row()
		if not self.should_exist:
			row = col.row()
			row.label(icon='NETWORK_DRIVE')
			row.prop(self, 'new_task_name', text='')

			col.separator(factor=1.0)

			box = col.box()
			row = box.row()

		row.prop(self, 'existing_task_id', text='')

	@property
	def value(
		self,
	) -> tuple[tdcloud.CloudTaskName, tdcloud.CloudFolder] | tdcloud.CloudTask | None:
		# Retrieve Folder
		## Authentication is presumed OK
		if (
			cloud_folder := tdcloud.TidyCloudFolders.folders().get(
				self.existing_folder_id
			)
		) is None:
			msg = "Selected folder doesn't exist (it was probably deleted elsewhere)"
			raise RuntimeError(msg)

		# No Tasks in Folder
		## The UI should set to "NONE" when there are no tasks in a folder
		if self.existing_task_id == 'NONE':
			return None

		# Retrieve Task
		if self.should_exist:
			if (
				cloud_task := tdcloud.TidyCloudTasks.tasks(cloud_folder).get(
					self.existing_task_id
				)
			) is None:
				msg = "Selected task doesn't exist (it was probably deleted elsewhere)"
				raise RuntimeError(msg)

			return cloud_task

		return (self.new_task_name, cloud_folder)


####################
# - Socket Configuration
####################
class Tidy3DCloudTaskSocketDef(base.SocketDef):
	socket_type: ct.SocketType = ct.SocketType.Tidy3DCloudTask

	should_exist: bool

	def init(self, bl_socket: Tidy3DCloudTaskBLSocket) -> None:
		bl_socket.should_exist = self.should_exist


####################
# - Blender Registration
####################
BL_REGISTER = [
	ReloadFolderList,
	Authenticate,
	Tidy3DCloudTaskBLSocket,
]
