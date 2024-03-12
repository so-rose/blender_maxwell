import typing as typ
import tempfile

import bpy
import pydantic as pyd
import tidy3d as td
import tidy3d.web as _td_web

from .....utils.auth_td_web import g_td_web, is_td_web_authed
from .. import base
from ... import contracts as ct

####################
# - Tidy3D Folder/Task Management
####################
TD_FOLDERS = None
## TODO: Keep this data serialized in each node, so it works offline and saves/loads correctly (then we can try/except when the network fails).
## - We should consider adding some kind of serialization-backed instance data to the node base class...
## - We could guard it behind a feature, 'use_node_data_store' for example.

def g_td_folders():
	global TD_FOLDERS
	
	if TD_FOLDERS is not None: return TD_FOLDERS
	
	# Populate Folders Cache & Return
	TD_FOLDERS = {
		cloud_folder.folder_name: None
		for cloud_folder in _td_web.core.task_core.Folder.list()
	}
	return TD_FOLDERS

def g_td_tasks(cloud_folder_name: str):
	global TD_FOLDERS
	
	# Retrieve Cached Tasks
	if (_tasks := TD_FOLDERS.get(cloud_folder_name)) is not None:
		return _tasks
	
	# Retrieve Cloud Folder (if exists)
	try:
		cloud_folder = _td_web.core.task_core.Folder.get(cloud_folder_name)
	except AttributeError as err:
		# Folder Doesn't Exist
		TD_FOLDERS = None
		return []
		
	# Return Tasks as List (also empty)
	if (tasks := cloud_folder.list_tasks()) is None:
		tasks = []
		
	# Populate Cloud-Folder Cache & Return
	TD_FOLDERS[cloud_folder_name] = [
		task
		for task in tasks
	]
	return TD_FOLDERS[cloud_folder_name]

class BlenderMaxwellRefreshTDFolderList(bpy.types.Operator):
	bl_idname = "blender_maxwell.refresh_td_folder_list"
	bl_label = "Refresh Tidy3D Folder List"
	bl_description = "Refresh the cached Tidy3D folder list"
	bl_options = {'REGISTER'}
	
	@classmethod
	def poll(cls, context):
		space = context.space_data
		return (
			space.type == 'NODE_EDITOR'
			and space.node_tree is not None
			and space.node_tree.bl_idname == "MaxwellSimTreeType"
			and is_td_web_authed()
		)

	def execute(self, context):
		global TD_FOLDERS
		
		TD_FOLDERS = None
		return {'FINISHED'}

class Tidy3DCloudTaskBLSocket(base.MaxwellSimSocket):
	socket_type = ct.SocketType.Tidy3DCloudTask
	bl_label = "Tidy3D Cloud Sim"
	
	####################
	# - Properties
	####################
	task_exists: bpy.props.BoolProperty(
		name="Cloud Task Should Exist",
		description="Whether or not the cloud task referred to should exist",
		default=False,
	)
	
	api_key: bpy.props.StringProperty(
		name="API Key",
		description="API Key for the Tidy3D Cloud",
		default="",
		options={"SKIP_SAVE"},
		subtype="PASSWORD",
	)
	
	existing_folder_name: bpy.props.EnumProperty(
		name="Folder of Cloud Tasks",
		description="An existing folder on the Tidy3D Cloud",
		items=lambda self, context: self.retrieve_folders(context),
		update=(lambda self, context: self.sync_prop("existing_folder_name", context)),
	)
	existing_task_id: bpy.props.EnumProperty(
		name="Existing Cloud Task",
		description="An existing task on the Tidy3D Cloud, within the given folder",
		items=lambda self, context: self.retrieve_tasks(context),
		update=(lambda self, context: self.sync_prop("existing_task_id", context)),
	)
	new_task_name: bpy.props.StringProperty(
		name="New Cloud Task Name",
		description="Name of a new task to submit to the Tidy3D Cloud",
		default="",
		update=(lambda self, context: self.sync_new_task(context)),
	)
	
	lock_nonauth_interface: bpy.props.BoolProperty(
		name="Lock the non-Auth Interface",
		description="Declares that the non-auth interface should be locked",
		default=False,
	)
	
	def retrieve_folders(self, context) -> list[tuple]:
		if not is_td_web_authed: return []
		## What if there are no folders?
		
		return [
			(
				folder_name,
				folder_name,
				folder_name,
			)
			for folder_name in g_td_folders()
		]
	
	def retrieve_tasks(self, context) -> list[tuple]:
		if not is_td_web_authed: return []
		if not (cloud_tasks := g_td_tasks(self.existing_folder_name)):
			return [("NONE", "None", "No tasks in folder")]
		
		return [
			(
				## Task ID
				task.task_id,
				
				## Task Dropdown Names
				" ".join([
					task.taskName,
					"(" + task.created_at.astimezone().strftime(
						'%y-%m-%d @ %H:%M %Z'
					) + ")",
				]),
				
				## Task Description
				{
					"draft": "Task has been uploaded, but not run",
					"initialized": "Task is initializing",
					"queued": "Task is queued for simulation",
					"preprocessing": "Task is pre-processing",
					"running": "Task is currently running",
					"postprocess": "Task is post-processing",
					"success": "Task ran successfully, costing {task.real_flex_unit} credits",
					"error": "Task ran, but an error occurred",
				}[task.status],
				
				## Status Icon
				{
					"draft": "SEQUENCE_COLOR_08",
					"initialized": "SHADING_SOLID",
					"queued": "SEQUENCE_COLOR_03",
					"preprocessing": "SEQUENCE_COLOR_02",
					"running": "SEQUENCE_COLOR_05",
					"postprocess": "SEQUENCE_COLOR_06",
					"success": "SEQUENCE_COLOR_04",
					"error": "SEQUENCE_COLOR_01",
				}[task.status],
				
				## Unique Number
				i,
			)
			for i, task in enumerate(
				sorted(cloud_tasks, key=lambda el: el.created_at, reverse=True)
			)
		]
	
	####################
	# - Task Sync Methods
	####################
	def sync_new_task(self, context):
		if self.new_task_name == "": return
		
		if self.new_task_name in {
			task.taskName
			for task in g_td_tasks(self.existing_folder_name)
		}:
			self.new_task_name = ""
		
		self.sync_prop("new_task_name", context)
	
	def sync_task_loaded(self, loaded_task_id: str | None):
		"""Called whenever a particular task has been loaded.
		
		This resets the 'new_task_name' (if any), sets the dropdown to the new loaded task (which must be in the already-selected folder) (or, if input is None, leaves the selection alone), locks the socket UI (though NEVER the API authentication interface), and declares that the specified task exists.
		"""
		global TD_FOLDERS
		## TODO: This doesn't work with a linked socket. It should.
		
		if not (TD_FOLDERS is None):
			TD_FOLDERS[self.existing_folder_name] = None
		
		if loaded_task_id is not None:
			self.existing_task_id = loaded_task_id
		
		self.new_task_name = ""
		self.lock_nonauth_interface = True
		self.task_exists = True
	
	def sync_task_status_change(self, running_task_id: str):
		global TD_FOLDERS
		## TODO: This doesn't work with a linked socket. It should.
		
		if not (TD_FOLDERS is None):
			TD_FOLDERS[self.existing_folder_name] = None
	
	def sync_task_released(self, specify_new_task: bool = False):
		## TODO: This doesn't work with a linked socket. It should.
		self.new_task_name = ""
		self.lock_nonauth_interface = False
		self.task_exists = not specify_new_task
	
	####################
	# - Socket UI
	####################
	def draw_label_row(self, row: bpy.types.UILayout, text: str):
		row.label(text=text)
		
		auth_icon = "CHECKBOX_HLT" if is_td_web_authed() else "CHECKBOX_DEHLT"
		row.operator(
			"blender_maxwell.refresh_td_auth",
			text="",
			icon=auth_icon,
		)
		
	def draw_value(self, col: bpy.types.UILayout) -> None:
		if is_td_web_authed():
			if self.lock_nonauth_interface: col.enabled = False
			else: col.enabled = True
			
			row = col.row()
			row.label(icon="FILE_FOLDER")
			row.prop(self, "existing_folder_name", text="")
			row.operator(
				BlenderMaxwellRefreshTDFolderList.bl_idname,
				text="",
				icon="FILE_REFRESH",
			)
			
			if not self.task_exists:
				row = col.row()
				row.label(icon="SEQUENCE_COLOR_04")
				row.prop(self, "new_task_name", text="")
			
			if self.task_exists:
				row = col.row()
			else:
				col.separator(factor=1.0)
				box = col.box()
				row = box.row()
				
			row.label(icon="NETWORK_DRIVE")
			row.prop(self, "existing_task_id", text="")
		
		else:
			col.enabled = True
			row = col.row()
			row.alignment="CENTER"
			row.label(text="Tidy3D API Key")
			
			row = col.row()
			row.prop(self, "api_key", text="")
	
	@property
	def value(self) -> str | None:
		if self.task_exists:
			if self.existing_task_id == "NONE": return None
			return self.existing_task_id
		
		return dict(
			task_name=self.new_task_name,
			folder_name=self.existing_folder_name,
		)

####################
# - Socket Configuration
####################
class Tidy3DCloudTaskSocketDef(pyd.BaseModel):
	socket_type: ct.SocketType = ct.SocketType.Tidy3DCloudTask
	
	task_exists: bool
	
	def init(self, bl_socket: Tidy3DCloudTaskBLSocket) -> None:
		bl_socket.task_exists = self.task_exists

####################
# - Blender Registration
####################
BL_REGISTER = [
	BlenderMaxwellRefreshTDFolderList,
	Tidy3DCloudTaskBLSocket,
]

