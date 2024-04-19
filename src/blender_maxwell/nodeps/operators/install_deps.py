import subprocess
import sys
from pathlib import Path

import bpy

from ... import contracts as ct
from ... import registration
from ..utils import pydeps, simple_logger

log = simple_logger.get(__name__)


class InstallPyDeps(bpy.types.Operator):
	bl_idname = ct.OperatorType.InstallPyDeps
	bl_label = 'Install BLMaxwell Python Deps'

	@classmethod
	def poll(cls, _: bpy.types.Context):
		return not pydeps.DEPS_OK

	####################
	# - Property: PyDeps Path
	####################
	bl__pydeps_path: bpy.props.StringProperty(
		default='',
	)

	@property
	def pydeps_path(self):
		return Path(bpy.path.abspath(self.bl__pydeps_path))

	@pydeps_path.setter
	def pydeps_path(self, path: Path) -> None:
		self.bl__pydeps_path = str(path.resolve())

	####################
	# - Property: requirements.lock
	####################
	bl__pydeps_reqlock_path: bpy.props.StringProperty(
		default='',
	)

	@property
	def pydeps_reqlock_path(self):
		return Path(bpy.path.abspath(self.bl__pydeps_reqlock_path))

	@pydeps_reqlock_path.setter
	def pydeps_reqlock_path(self, path: Path) -> None:
		self.bl__pydeps_reqlock_path = str(path.resolve())

	####################
	# - Execution
	####################
	def execute(self, _: bpy.types.Context):
		log.info(
			'Running Install PyDeps w/requirements.txt (%s) to path: %s',
			self.pydeps_reqlock_path,
			self.pydeps_path,
		)

		# Create the Addon-Specific Folder (if Needed)
		## It MUST, however, have a parent already
		self.pydeps_path.mkdir(parents=False, exist_ok=True)

		# Determine Path to Blender's Bundled Python
		## bpy.app.binary_path_python was deprecated in 2.91.
		## sys.executable points to the correct bundled Python.
		## See <https://developer.blender.org/docs/release_notes/2.91/python_api/>
		python_exec = Path(sys.executable)

		# Install Deps w/Bundled pip
		try:
			cmdline = [
				str(python_exec),
				'-m',
				'pip',
				'install',
				'-r',
				str(self.pydeps_reqlock_path),
				'--target',
				str(self.pydeps_path),
			]
			log.info(
				'Running pip w/cmdline: %s',
				' '.join(cmdline),
			)
			subprocess.check_call(cmdline)
		except subprocess.CalledProcessError:
			log.exception('Failed to install PyDeps')
			return {'CANCELLED'}

		# Report PyDeps Changed
		ct.addon.prefs().on_addon_pydeps_changed()
		return {'FINISHED'}


####################
# - Blender Registration
####################
BL_REGISTER = [
	InstallPyDeps,
]
BL_HOTKEYS = []
