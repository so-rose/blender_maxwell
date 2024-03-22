"""Defines a sane interface to the Tidy3D cloud, as constructed by reverse-engineering the official open-source `tidy3d` client library.
- SimulationTask: <https://github.com/flexcompute/tidy3d/blob/453055e89dcff6d619597120b47817e996f1c198/tidy3d/web/core/task_core.py>
- Tidy3D Stub: <https://github.com/flexcompute/tidy3d/blob/453055e89dcff6d619597120b47817e996f1c198/tidy3d/web/api/tidy3d_stub.py>
"""

import datetime as dt
import functools
import typing as typ
from dataclasses import dataclass

import tidy3d as td
import tidy3d.web as td_web

CloudFolderID = str
CloudFolderName = str
CloudFolder = td_web.core.task_core.Folder

CloudTaskID = str
CloudTaskName = str
CloudTask = td_web.core.task_core.SimulationTask

FileUploadCallback = typ.Callable[[float], None]
## Takes "uploaded bytes" as argument.

####################
# - Module-Level Globals
####################
IS_ONLINE = False
IS_AUTHENTICATED = False


def set_online():
	global IS_ONLINE  # noqa: PLW0603
	IS_ONLINE = True


def set_offline():
	global IS_ONLINE  # noqa: PLW0603
	IS_ONLINE = False


####################
# - Cloud Authentication
####################
def check_authentication() -> bool:
	global IS_AUTHENTICATED  # noqa: PLW0603

	# Check Previous Authentication
	## If we authenticated once, we presume that it'll work again.
	## TODO: API keys can change... It would just look like "offline" for now.
	if IS_AUTHENTICATED:
		return True

	api_key = td_web.core.http_util.api_key()
	if api_key is not None:
		try:
			td_web.test()
			set_online()
		except td.exceptions.WebError:
			set_offline()
			return False

		IS_AUTHENTICATED = True
		return True

	return False


def authenticate_with_api_key(api_key: str) -> bool:
	td_web.configure(api_key)
	return check_authentication()


####################
# - Cloud Folder
####################
class TidyCloudFolders:
	cache_folders: dict[CloudFolderID, CloudFolder] | None = None

	####################
	# - Folders
	####################
	@classmethod
	def folders(cls) -> dict[CloudFolderID, CloudFolder]:
		"""Get all cloud folders as a dict, indexed by ID."""
		if cls.cache_folders is not None:
			return cls.cache_folders

		try:
			cloud_folders = td_web.core.task_core.Folder.list()
			set_online()
		except td.exceptions.WebError as ex:
			set_offline()
			msg = 'Tried to get cloud folders, but cannot connect to cloud'
			raise RuntimeError(msg) from ex

		folders = {
			cloud_folder.folder_id: cloud_folder for cloud_folder in cloud_folders
		}
		cls.cache_folders = folders
		return folders

	@classmethod
	def mk_folder(cls, folder_name: CloudFolderName) -> CloudFolder:
		"""Create a cloud folder, raising an exception if it exists."""
		folders = cls.update_folders()
		if folder_name not in {
			cloud_folder.folder_name for cloud_folder in folders.values()
		}:
			try:
				cloud_folder = td_web.core.task_core.Folder.create(folder_name)
				set_online()
			except td.exceptions.WebError as ex:
				set_offline()
				msg = 'Tried to create cloud folder, but cannot connect to cloud'
				raise RuntimeError(msg) from ex

			if cls.cache_folders is None:
				cls.cache_folders = {}
			cls.cache_folders[cloud_folder.folder_id] = cloud_folder
			return cloud_folder

		msg = f"Cannot create cloud folder: Folder '{folder_name}' already exists"
		raise ValueError(msg)

	@classmethod
	def update_folders(cls) -> dict[CloudFolderID, CloudFolder]:
		"""Get all cloud folders as a dict, forcing a re-check with the web service."""
		cls.cache_folders = None
		return cls.folders()

	## TODO: Support removing folders. Unsure of the semantics (does it recursively delete tasks too?)


####################
# - Cloud Task
####################
@dataclass
class CloudTaskInfo:
	"""Toned-down, simplified `dataclass` variant of TaskInfo.

	See TaskInfo for more: <https://github.com/flexcompute/tidy3d/blob/453055e89dcff6d619597120b47817e996f1c198/tidy3d/web/core/task_info.py>)
	"""

	task_name: str
	status: str
	created_at: dt.datetime

	cost_est: typ.Callable[[], float | None]
	run_info: typ.Callable[[], tuple[float | None, float | None] | None]

	# Timing
	completed_at: dt.datetime | None = None  ## completedAt

	# Cost
	cost_real: float | None = None  ## realCost

	# Sim Properties
	task_type: str | None = None  ## solverVersion
	version_solver: str | None = None  ## solverVersion
	callback_url: str | None = None  ## callbackUrl


class TidyCloudTasks:
	"""Greatly simplifies working with Tidy3D Tasks in the Cloud, specifically, via the lowish-level `tidy3d.web.core.task_core.SimulationTask` object.

	In particular, cache mechanics ensure that web-requests are only made when absolutely needed.
	This greatly improves performance in ex. UI functions.
	In particular, `update_task` updates only one task with a single request.

	Of particular note are the `SimulationTask` methods that are not abstracted:
	- `cloud_task.taskName`: Undocumented, but it works (?)
	- `cloud_task.submit()`: Starts the running of a drafted task.
	- `cloud_task.real_flex_unit`: `None` until available. Just repeat `update_task` until not None.
	- `cloud_task.get_running_info()`: GETs % and field-decay of a running task.
	- `cloud_task.get_log(path)`: GET the run log. Remember to use `NamedTemporaryFile` if a stringified log is desired.
	"""

	cache_tasks: typ.ClassVar[dict[CloudTaskID, CloudTask]] = {}
	cache_folder_tasks: typ.ClassVar[dict[CloudFolderID, set[CloudTaskID]]] = {}
	cache_task_info: typ.ClassVar[dict[CloudTaskID, CloudTaskInfo]] = {}

	@classmethod
	def clear_cache(cls):
		cls.cache_tasks = {}

	####################
	# - Task Getters
	####################
	@classmethod
	def task(cls, task_id: CloudTaskID) -> CloudTask | None:
		return cls.cache_tasks.get(task_id)

	@classmethod
	def task_info(cls, task_id: CloudTaskID) -> CloudTaskInfo | None:
		return cls.cache_task_info.get(task_id)

	@classmethod
	def tasks(cls, cloud_folder: CloudFolder) -> dict[CloudTaskID, CloudTask]:
		"""Get all cloud tasks within a particular cloud folder as a set."""
		# Retrieve Cached Tasks
		if (task_ids := cls.cache_folder_tasks.get(cloud_folder.folder_id)) is not None:
			return {task_id: cls.cache_tasks[task_id] for task_id in task_ids}

		# Retrieve Tasks by-Folder
		try:
			folder_tasks = cloud_folder.list_tasks()
			set_online()
		except td.exceptions.WebError as ex:
			set_offline()
			msg = 'Tried to get tasks of a cloud folder, but cannot access cloud'
			raise RuntimeError(msg) from ex

		# No Tasks: Empty Set
		if folder_tasks is None:
			cls.cache_folder_tasks[cloud_folder.folder_id] = set()
			return {}

		# Populate Caches
		## Direct Task Cache
		cloud_tasks = {cloud_task.task_id: cloud_task for cloud_task in folder_tasks}
		cls.cache_tasks |= cloud_tasks

		## Task Info Cache
		for task_id, cloud_task in cloud_tasks.items():
			cls.cache_task_info[task_id] = CloudTaskInfo(
				task_name=cloud_task.taskName,
				status=cloud_task.status,
				created_at=cloud_task.created_at,
				cost_est=functools.partial(td_web.estimate_cost, cloud_task.task_id),
				run_info=cloud_task.get_running_info,
				callback_url=cloud_task.callback_url,
			)

		## Task by-Folder Cache
		cls.cache_folder_tasks[cloud_folder.folder_id] = set(cloud_tasks)

		return cloud_tasks

	####################
	# - Task Create/Delete
	####################
	@classmethod
	def mk_task(
		cls,
		task_name: CloudTaskName,
		cloud_folder: CloudFolder,
		sim: td.Simulation,
		upload_progress_cb: FileUploadCallback | None = None,
		verbose: bool = True,
	) -> CloudTask:
		"""Creates a `CloudTask` of the given `td.Simulation`.

		Presume that `sim.validate_pre_upload()` has already been run, so that the simulation is good to go.
		"""
		# Create "Stub"
		## Minimal Tidy3D object that can be turned into a file for upload
		## Has "type" in {"Simulation", "ModeSolver", "HeatSimulation"}
		stub = td_web.api.tidy3d_stub.Tidy3dStub(simulation=sim)

		# Create Cloud Task
		## So far, this is a boring, empty task with no data
		## May overlay by name with other tasks - then makes a new "version"
		try:
			cloud_task = td_web.core.task_core.SimulationTask.create(
				task_type=stub.get_type(),
				task_name=task_name,
				folder_name=cloud_folder.folder_name,
			)
			set_online()
		except td.exceptions.WebError as ex:
			set_offline()
			msg = 'Tried to create cloud task, but cannot access cloud'
			raise RuntimeError(msg) from ex

		# Upload Simulation to Cloud Task
		if upload_progress_cb is not None:
			raise NotImplementedError
		try:
			cloud_task.upload_simulation(
				stub,
				verbose=verbose,
				# progress_callback=upload_progress_cb,
			)
			set_online()
		except td.exceptions.WebError as ex:
			set_offline()
			msg = 'Tried to upload simulation to cloud task, but cannot access cloud'
			raise RuntimeError(msg) from ex

		# Populate Caches
		## Direct Task Cache
		cls.cache_tasks[cloud_task.task_id] = cloud_task

		## Task Info Cache
		cls.cache_task_info[cloud_task.task_id] = CloudTaskInfo(
			task_name=cloud_task.taskName,
			status=cloud_task.status,
			created_at=cloud_task.created_at,
			cost_est=functools.partial(td_web.estimate_cost, cloud_task.task_id),
			run_info=cloud_task.get_running_info,
			callback_url=cloud_task.callback_url,
		)

		## Task by-Folder Cache
		if cls.cache_folder_tasks.get(cloud_task.folder_id):
			cls.cache_folder_tasks[cloud_task.folder_id].add(cloud_task.task_id)
		else:
			cls.cache_folder_tasks[cloud_task.folder_id] = {cloud_task.task_id}

		return cloud_task

	####################
	# - Task Update/Delete
	####################
	@classmethod
	def rm_task(
		cls,
		cloud_task: CloudTask,
	) -> CloudTask:
		"""Deletes a cloud task."""
		## TODO: Abort first?
		task_id = cloud_task.task_id
		folder_id = cloud_task.folder_id
		try:
			cloud_task.delete()
			set_online()
		except td.exceptions.WebError as ex:
			set_offline()
			msg = 'Tried to delete cloud task, but cannot access cloud'
			raise RuntimeError(msg) from ex

		# Populate Caches
		## Direct Task Cache
		cls.cache_tasks.pop(task_id, None)

		## Task Info Cache
		cls.cache_task_info.pop(task_id, None)

		## Task by-Folder Cache
		cls.cache_folder_tasks[folder_id].remove(task_id)

	@classmethod
	def update_task(cls, cloud_task: CloudTask) -> CloudTask:
		"""Updates the CloudTask to the latest ex. status attributes."""
		# BUG: td_web.core.task_core.SimulationTask.get(task_id) doesn't return the `created_at` field.
		## Therefore, we unfortunately need to get all tasks for the folder ID just to update one.

		# Retrieve Folder
		task_id = cloud_task.task_id
		folder_id = cloud_task.folder_id
		cloud_folder = TidyCloudFolders.folders()[folder_id]

		# Repopulate All Caches
		## By deleting the folder ID, all tasks within will be reloaded
		del cls.cache_folder_tasks[folder_id]

		return cls.tasks(cloud_folder)[task_id]

	@classmethod
	def update_tasks(cls, folder_id: CloudFolderID) -> dict[CloudTaskID, CloudTask]:
		"""Updates the CloudTask to the latest ex. status attributes."""
		# BUG: td_web.core.task_core.SimulationTask.get(task_id) doesn't return the `created_at` field.
		## Therefore, we unfortunately need to get all tasks for the folder ID just to update one.

		# Retrieve Folder
		cloud_folder = TidyCloudFolders.folders()[folder_id]

		# Repopulate All Caches
		## By deleting the folder ID, all tasks within will be reloaded
		del cls.cache_folder_tasks[folder_id]

		return {
			task_id: cls.tasks(cloud_folder)[task_id]
			for task_id in cls.cache_folder_tasks[folder_id]
		}

	@classmethod
	def abort_task(cls, cloud_task: CloudTask) -> CloudTask:
		"""Aborts a running CloudTask to the latest ex. status attributes."""
		## TODO: Check status?
		new_cloud_task = cls.update_task(cloud_task)
		try:
			new_cloud_task.abort()
			set_online()
		except td.exceptions.WebError as ex:
			set_offline()
			msg = 'Tried to abort cloud task, but cannot access cloud'
			raise RuntimeError(msg) from ex

		return cls.update_task(cloud_task)
