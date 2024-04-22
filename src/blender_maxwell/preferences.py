import logging
from pathlib import Path

import bpy

from . import contracts as ct
from . import registration
from .nodeps.operators import install_deps, uninstall_deps
from .nodeps.utils import pip_process, pydeps, simple_logger

log = simple_logger.get(__name__)


####################
# - Preferences
####################
class BLMaxwellAddonPrefs(bpy.types.AddonPreferences):
	"""Manages user preferences and settings for the Blender Maxwell addon.

	Unfortunately, many of the niceities based on dependencies (ex. `bl_cache.BLField`) aren't available here.


	Attributes:
		bl_idname: Matches `ct.addon.NAME`.
		use_default_pydeps_path: Whether to use the default PyDeps path
	"""

	bl_idname = ct.addon.NAME

	####################
	# - Properties
	####################
	# PyDeps Default Path
	use_default_pydeps_path: bpy.props.BoolProperty(
		name='Use Default PyDeps Path',
		description='Whether to use the default PyDeps path',
		default=True,
		update=lambda self, context: self.on_addon_pydeps_changed(context),
	)

	# PyDeps Path
	bl__pydeps_path: bpy.props.StringProperty(
		name='Addon PyDeps Path',
		description='Path to Addon Python Dependencies',
		subtype='FILE_PATH',
		default=str(ct.addon.DEFAULT_PATH_DEPS),
		update=lambda self, _: self.on_addon_pydeps_changed(),
	)

	cache__backup_pydeps_path: bpy.props.StringProperty(
		default=str(ct.addon.DEFAULT_PATH_DEPS),
	)

	@property
	def pydeps_path(self) -> Path:
		if self.use_default_pydeps_path:
			return ct.addon.DEFAULT_PATH_DEPS

		return Path(bpy.path.abspath(self.bl__pydeps_path))

	@pydeps_path.setter
	def pydeps_path(self, path: Path) -> None:
		if not self.use_default_pydeps_path:
			self.bl__pydeps_path = str(path.resolve())
		else:
			msg = f'Can\'t set "pydeps_path" to {path} while "use_default_pydeps_path" is "True"'
			raise ValueError(msg)

	# Logging
	## Console Logging
	use_log_console: bpy.props.BoolProperty(
		name='Log to Console',
		description='Whether to use the console for addon logging',
		default=True,
		update=lambda self, _: self.on_addon_logging_changed(),
	)
	log_level_console: bpy.props.EnumProperty(
		name='Console Log Level',
		description='Level of addon logging to expose in the console',
		items=[
			('DEBUG', 'Debug', 'Debug'),
			('INFO', 'Info', 'Info'),
			('WARNING', 'Warning', 'Warning'),
			('ERROR', 'Error', 'Error'),
			('CRITICAL', 'Critical', 'Critical'),
		],
		default='DEBUG',
		update=lambda self, _: self.on_addon_logging_changed(),
	)
	## TODO: Derive default from BOOTSTRAP_LOG_LEVEL

	## File Logging
	use_log_file: bpy.props.BoolProperty(
		name='Log to File',
		description='Whether to use a file for addon logging',
		default=True,
		update=lambda self, _: self.on_addon_logging_changed(),
	)
	log_level_file: bpy.props.EnumProperty(
		name='File Log Level',
		description='Level of addon logging to expose in the file',
		items=[
			('DEBUG', 'Debug', 'Debug'),
			('INFO', 'Info', 'Info'),
			('WARNING', 'Warning', 'Warning'),
			('ERROR', 'Error', 'Error'),
			('CRITICAL', 'Critical', 'Critical'),
		],
		default='DEBUG',
		update=lambda self, _: self.on_addon_logging_changed(),
	)

	bl__log_file_path: bpy.props.StringProperty(
		name='Log Path',
		description='Path to the Addon Log File',
		subtype='FILE_PATH',
		default=str(ct.addon.DEFAULT_LOG_PATH),
		update=lambda self, _: self.on_addon_logging_changed(),
	)

	@property
	def log_file_path(self) -> Path:
		return Path(bpy.path.abspath(self.bl__log_file_path))

	@log_file_path.setter
	def log_file_path(self, path: Path) -> None:
		self.bl__log_file_path = str(path.resolve())

	####################
	# - Events: Properties Changed
	####################
	def on_addon_logging_changed(
		self, single_logger_to_setup: logging.Logger | None = None
	) -> None:
		"""Configure one, or all, active addon logger(s).

		Parameters:
			single_logger_to_setup: When set, only this logger will be setup.
				Otherwise, **all addon loggers will be setup**.
		"""
		if pydeps.DEPS_OK:
			with pydeps.importable_addon_deps(self.pydeps_path):
				from blender_maxwell.utils import logger
		else:
			logger = simple_logger

		# Retrieve Configured Log Levels
		log_level_console = logger.LOG_LEVEL_MAP[self.log_level_console]
		log_level_file = logger.LOG_LEVEL_MAP[self.log_level_file]

		log_setup_kwargs = {
			'console_level': log_level_console if self.use_log_console else None,
			'file_path': self.log_file_path if self.use_log_file else None,
			'file_level': log_level_file,
		}

		# Sync Single Logger / All Loggers
		if single_logger_to_setup is not None:
			logger.update_logger(
				logger.console_handler,
				logger.file_handler,
				single_logger_to_setup,
				**log_setup_kwargs,
			)
		else:
			log.info('Re-Configuring All Loggers')
			logger.update_all_loggers(
				logger.console_handler,
				logger.file_handler,
				**log_setup_kwargs,
			)

	def on_addon_pydeps_changed(self, show_popup_if_deps_invalid: bool = False) -> None:
		"""Checks if the Python dependencies are valid, and runs any delayed setup (inclusing `ct.BLClass` registrations) in response.

		Notes:
			**The addon does not load until this method allows it**.

		Parameters:
			show_popup_if_deps_invalid: If True, a failed dependency check will `invoke()` the operator `ct.OperatorType.ManagePyDeps`, which is a popup that guides the user through
				**NOTE**: Must be called after addon registration.

		Notes:
			Run by `__init__.py` after registering a barebones addon (including this class), and after queueing a delayed registration.
		"""
		if pydeps.check_pydeps(ct.addon.PATH_REQS, self.pydeps_path):
			# Re-Sync Loggers
			## We can now upgrade all loggers to the fancier loggers.
			for _log in simple_logger.simple_loggers():
				log.debug('Upgrading Logger (%s)', str(_log))
				self.on_addon_logging_changed(single_logger_to_setup=_log)
			simple_logger.clear_simple_loggers()

			# Run Registrations Waiting on DEPS_SATISFIED
			## Since the deps are OK, we can now register the whole addon.
			if (
				registration.BLRegisterEvent.DepsSatisfied
				in registration.DELAYED_REGISTRATIONS
			):
				registration.run_delayed_registration(
					registration.BLRegisterEvent.DepsSatisfied,
					self.pydeps_path,
				)

		elif show_popup_if_deps_invalid:
			ct.addon.operator(
				ct.OperatorType.ManagePyDeps,
				'INVOKE_DEFAULT',
				bl__pydeps_path=str(self.pydeps_path),
			)
		## TODO: else:
		## TODO: Can we 'downgrade' the loggers back to simple loggers?
		## TODO: Can we undo the delayed registration?
		## TODO: Do we need the fancy pants sys.modules handling for all this?

	####################
	# - UI
	####################
	def draw(self, _: bpy.types.Context) -> None:
		layout = self.layout
		num_pydeps_issues = len(pydeps.DEPS_ISSUES)

		####################
		# - Logging
		####################
		# Box w/Split: Log Level
		box = layout.box()
		row = box.row()
		row.alignment = 'CENTER'
		row.label(text='Logging')
		split = box.split(factor=0.5)

		## Split Col: Console Logging
		col = split.column()
		row = col.row()
		row.prop(self, 'use_log_console', toggle=True)

		row = col.row()
		row.enabled = self.use_log_console
		row.prop(self, 'log_level_console')

		## Split Col: File Logging
		col = split.column()
		row = col.row()
		row.prop(self, 'use_log_file', toggle=True)

		row = col.row()
		row.enabled = self.use_log_file
		row.prop(self, 'bl__log_file_path')

		row = col.row()
		row.enabled = self.use_log_file
		row.prop(self, 'log_level_file')

		####################
		# - Dependencies
		####################
		# Box: Dependency Status
		box = layout.box()
		row = box.row(align=True)
		row.alignment = 'CENTER'
		row.label(text='Python Dependencies')

		## Row: Toggle Default PyDeps Path
		row = box.row(align=True)
		row.enabled = not pydeps.DEPS_OK
		row.prop(
			self,
			'use_default_pydeps_path',
			text='Use Default PyDeps Install Path',
			toggle=True,
		)

		## Row: Current PyDeps Path
		row = box.row(align=True)
		row.enabled = not pydeps.DEPS_OK and not self.use_default_pydeps_path
		row.prop(self, 'bl__pydeps_path', text='PyDeps Install Path')

		## Row: More Information Panel
		col = box.column(align=True)
		header, panel = col.panel('pydeps_issues', default_closed=True)
		header.label(text=f'Show Conflicts ({num_pydeps_issues})')
		if panel is not None:
			grid = panel.grid_flow()
			for issue in pydeps.DEPS_ISSUES:
				grid.label(text=issue)

		## Row: Install
		row = box.row(align=True)
		op = row.operator(
			install_deps.InstallPyDeps.bl_idname,
			text='Install PyDeps',
		)
		op.bl__pydeps_path = str(self.pydeps_path)
		op.bl__pydeps_reqlock_path = str(ct.addon.PATH_REQS)

		## Row: Uninstall
		row = box.row(align=True)
		op = row.operator(
			uninstall_deps.UninstallPyDeps.bl_idname,
			text='Uninstall PyDeps',
		)
		op.bl__pydeps_path = str(self.pydeps_path)

		## Row: Deps Install Progress
		row = box.row()
		num_req_deplocks = len(pydeps.DEPS_REQ_DEPLOCKS)
		if pydeps.DEPS_OK:
			row.progress(
				text=f'{num_req_deplocks}/{num_req_deplocks} Installed',
				factor=1.0,
			)
		elif pip_process.PROGRESS is not None:
			row.progress(
				text='/'.join(pip_process.PROGRESS_FRAC) + ' Installed',
				factor=float(pip_process.PROGRESS),
			)
		else:
			row.progress(
				text=f'0/{num_req_deplocks} Installed',
				factor=0.0,
			)


####################
# - Blender Registration
####################
BL_REGISTER = [
	BLMaxwellAddonPrefs,
]
