import subprocess
import sys
from pathlib import Path

import bpy

from ... import contracts as ct
from ..utils import pip_process, pydeps, simple_logger

log = simple_logger.get(__name__)


class InstallPyDeps(bpy.types.Operator):
	bl_idname = ct.OperatorType.InstallPyDeps
	bl_label = 'Install BLMaxwell Python Deps'

	@classmethod
	def poll(cls, _: bpy.types.Context):
		return not pip_process.is_loaded() and not pydeps.DEPS_OK

	####################
	# - Property: PyDeps Path
	####################
	_timer = None

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
	def execute(self, context: bpy.types.Context):
		if pip_process.is_loaded():
			self.report(
				{'ERROR'},
				'A PyDeps installation is already running. Please wait for it to complete.',
			)
			return {'FINISHED'}

		log.info(
			'Installing PyDeps to path: %s',
			str(self.pydeps_path),
		)

		# Create the Addon-Specific Folder (if Needed)
		## It MUST, however, have a parent already
		self.pydeps_path.mkdir(parents=False, exist_ok=True)

		# Run Pip Install
		pip_process.run(ct.addon.PATH_REQS, self.pydeps_path, ct.addon.PIP_INSTALL_LOG)

		# Set Timer
		self._timer = context.window_manager.event_timer_add(
			0.25, window=context.window
		)
		context.window_manager.modal_handler_add(self)

		return {'RUNNING_MODAL'}

	def modal(
		self, context: bpy.types.Context, event: bpy.types.Event
	) -> ct.BLOperatorStatus:
		# Non-Timer Event: Do Nothing
		if event.type != 'TIMER':
			return {'PASS_THROUGH'}

		# No Process: Very Bad!
		if not pip_process.is_loaded():
			msg = 'Pip process was removed elsewhere than "install_deps" modal operator'
			raise RuntimeError(msg)

		# Not Running: Done!
		if not pip_process.is_running():
			# Report Result
			if pip_process.returncode() == 0:
				self.report({'INFO'}, 'PyDeps installation succeeded.')
			else:
				self.report(
					{'ERROR'},
					f'PyDeps installation returned status code: {pip_process.returncode()}. Please see the addon preferences, or the pip installation logs at: {ct.addon.PIP_INSTALL_LOG}',
				)

			# Reset Process and Timer
			pip_process.reset()
			context.window_manager.event_timer_remove(self._timer)

			# Mark PyDeps Changed
			ct.addon.prefs().on_addon_pydeps_changed()

			return {'FINISHED'}

		if ct.addon.PIP_INSTALL_LOG.is_file():
			pip_process.update_progress(ct.addon.PIP_INSTALL_LOG)
			context.area.tag_redraw()
		return {'PASS_THROUGH'}

	def cancel(self, context: bpy.types.Context):
		# Kill / Reset Process and Delete Event Timer
		pip_process.kill()
		pip_process.reset()
		context.window_manager.event_timer_remove(self._timer)

		return {'CANCELLED'}


####################
# - Blender Registration
####################
BL_REGISTER = [
	InstallPyDeps,
]
BL_HOTKEYS = []
