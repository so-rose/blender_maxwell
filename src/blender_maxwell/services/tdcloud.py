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

"""Defines a sane interface to the Tidy3D cloud, as constructed by reverse-engineering the official open-source `tidy3d` client library.

- SimulationTask: <https://github.com/flexcompute/tidy3d/blob/453055e89dcff6d619597120b47817e996f1c198/tidy3d/web/core/task_core.py>
- Tidy3D Stub: <https://github.com/flexcompute/tidy3d/blob/453055e89dcff6d619597120b47817e996f1c198/tidy3d/web/api/tidy3d_stub.py>
"""

import datetime as dt
import functools
import tempfile
import typing as typ
import urllib
from dataclasses import dataclass
from pathlib import Path

import bpy
import tidy3d as td
import tidy3d.web as td_web

from blender_maxwell.utils import logger

log = logger.get(__name__)

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


def check_online() -> bool:
	global IS_ONLINE  # noqa: PLW0603
	log.info('Checking Internet Connection...')

	try:
		urllib.request.urlopen(
			'https://docs.flexcompute.com/projects/tidy3d/en/latest/index.html',
			timeout=2,
		)
	except:  # noqa: E722
		log.info('Internet is currently offline')
		IS_ONLINE = False
		return False
	else:
		log.info('Internet connection is working')
		IS_ONLINE = True
		return True


####################
# - Cloud Authentication
####################
def check_authentication() -> bool:
	global IS_AUTHENTICATED  # noqa: PLW0603
	log.info('Checking Tidy3D Authentication...')

	api_key = td_web.core.http_util.api_key()
	if api_key is not None:
		log.info('Found stored Tidy3D API key')
		try:
			td_web.test()
		except td.exceptions.WebError:
			set_offline()
			log.info('Authenticated to Tidy3D cloud')
			return False
		else:
			set_online()
			log.info('Authenticated to Tidy3D cloud')

		IS_AUTHENTICATED = True
		return True

	log.info('Tidy3D API key is missing')
	return False


def authenticate_with_api_key(api_key: str) -> bool:
	td_web.configure(api_key)
	return check_authentication()


TD_CONFIG = Path(td_web.cli.constants.CONFIG_FILE)

## TODO: Robustness is key - internet might be down.
## -> I'm not a huge fan of the max 2sec startup time burden
## -> I also don't love "calling" Tidy3D on startup, privacy-wise
if TD_CONFIG.is_file() and check_online():
	check_authentication()


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
		log.info('Retrieved Folders: %s', str(cls.cache_folders))
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
	"""Toned-down `dataclass` variant of `tidy3d`'s TaskInfo.

	See TaskInfo for more: <https://github.com/flexcompute/tidy3d/blob/453055e89dcff6d619597120b47817e996f1c198/tidy3d/web/core/task_info.py>)
	"""

	task_id: str
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

	def disk_cache_path(self, addon_cache: Path) -> Path:
		"""Compute an appropriate location for caching simulations downloaded from the internet, unique to each task ID.

		Arguments:
			task_id: The ID of the Tidy3D cloud task.
		"""
		(addon_cache / self.task_id).mkdir(exist_ok=True)
		return addon_cache / self.task_id / 'sim_data.hdf5'


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
				task_id=task_id,
				task_name=cloud_task.taskName,
				status=cloud_task.status,
				created_at=cloud_task.created_at,
				cost_est=functools.partial(td_web.estimate_cost, cloud_task.task_id),
				run_info=cloud_task.get_running_info,
				callback_url=cloud_task.callback_url,
			)

		## Task by-Folder Cache
		cls.cache_folder_tasks[cloud_folder.folder_id] = set(cloud_tasks)

		log.info(
			'Retrieved Tasks (folder="%s"): %s)',
			cloud_folder.folder_id,
			str(set(cloud_tasks)),
		)
		return cloud_tasks

	####################
	# - Task Download
	####################
	@classmethod
	def download_task_sim_data(
		cls, cloud_task: CloudTask, download_sim_path: Path | None = None
	) -> td.SimulationData:
		# Expose Path for SimData Download
		if download_sim_path is None:
			with tempfile.NamedTemporaryFile(delete=False) as f:
				_path_tmp = Path(f.name)
				_path_tmp.rename(f.name + '.hdf5.gz')
				path_sim = Path(f.name)
		else:
			path_sim = download_sim_path

		# Get Sim Data (from file and/or download)
		if path_sim.is_file():
			log.info('Loading Cloud Task "%s" from "%s"', cloud_task.task_id, path_sim)
			sim_data = td.SimulationData.from_file(str(path_sim))
		else:
			log.info(
				'Downloading & Loading Cloud Task "%s" to "%s"',
				cloud_task.task_id,
				path_sim,
			)
			sim_data = td_web.api.webapi.load(
				cloud_task.task_id,
				path=str(path_sim),
				replace_existing=True,
				verbose=True,
			)

		# Delete Temporary File (if used)
		if download_sim_path is None:
			Path(path_sim).unlink()

		return sim_data

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
			task_id=cloud_task.task_id,
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
		cls.cache_folder_tasks.pop(folder_id, None)

		return dict(cls.tasks(cloud_folder).items())

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


####################
# - Blender UI Integration
####################
def draw_cloud_status(layout: bpy.types.UILayout) -> None:
	"""Draw up-to-date information about the connection to the Tidy3D cloud to a Blender UI."""
	# Connection Info
	auth_icon = 'CHECKBOX_HLT' if IS_AUTHENTICATED else 'CHECKBOX_DEHLT'
	conn_icon = 'CHECKBOX_HLT' if IS_ONLINE else 'CHECKBOX_DEHLT'

	row = layout.row()
	row.alignment = 'CENTER'
	row.label(text='Cloud Status')

	box = layout.box()
	split = box.split(factor=0.85)

	col = split.column(align=False)
	col.label(text='Authed')
	col.label(text='Connected')

	col = split.column(align=False)
	col.label(icon=auth_icon)
	col.label(icon=conn_icon)
