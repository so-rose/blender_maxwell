import logging
from pathlib import Path

import bpy

from . import info, registration
from .nodeps.operators import install_deps, uninstall_deps
from .nodeps.utils import pydeps, simple_logger

####################
# - Constants
####################
log = simple_logger.get(__name__)


####################
# - Preferences
####################
class BLMaxwellAddonPrefs(bpy.types.AddonPreferences):
	"""Manages user preferences and settings for the Blender Maxwell addon."""

	bl_idname = info.ADDON_NAME  ## MUST match addon package name

	####################
	# - Properties
	####################
	# Use of Default PyDeps Path
	use_default_pydeps_path: bpy.props.BoolProperty(
		name='Use Default PyDeps Path',
		description='Whether to use the default PyDeps path',
		default=True,
		update=lambda self, context: self.sync_use_default_pydeps_path(context),
	)
	cache__pydeps_path_while_using_default: bpy.props.StringProperty(
		name='Cached Addon PyDeps Path',
		default=(_default_pydeps_path := str(info.DEFAULT_PATH_DEPS)),
	)

	# Custom PyDeps Path
	bl__pydeps_path: bpy.props.StringProperty(
		name='Addon PyDeps Path',
		description='Path to Addon Python Dependencies',
		subtype='FILE_PATH',
		default=_default_pydeps_path,
		update=lambda self, _: self.sync_pydeps_path(),
	)
	cache__backup_pydeps_path: bpy.props.StringProperty(
		name='Previous Addon PyDeps Path',
		default=_default_pydeps_path,
	)

	# Log Settings
	use_log_console: bpy.props.BoolProperty(
		name='Log to Console',
		description='Whether to use the console for addon logging',
		default=True,
		update=lambda self, _: self.sync_addon_logging(),
	)
	bl__log_level_console: bpy.props.EnumProperty(
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
		update=lambda self, _: self.sync_addon_logging(),
	)

	use_log_file: bpy.props.BoolProperty(
		name='Log to File',
		description='Whether to use a file for addon logging',
		default=True,
		update=lambda self, _: self.sync_addon_logging(),
	)
	bl__log_file_path: bpy.props.StringProperty(
		name='Log Path',
		description='Path to the Addon Log File',
		subtype='FILE_PATH',
		default=str(info.DEFAULT_LOG_PATH),
		update=lambda self, _: self.sync_addon_logging(),
	)
	bl__log_level_file: bpy.props.EnumProperty(
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
		update=lambda self, _: self.sync_addon_logging(),
	)

	# TODO: LOGGING SETTINGS

	####################
	# - Property Methods
	####################
	@property
	def pydeps_path(self) -> Path:
		return Path(bpy.path.abspath(self.bl__pydeps_path))

	@pydeps_path.setter
	def pydeps_path(self, value: Path) -> None:
		self.bl__pydeps_path = str(value.resolve())

	@property
	def log_path(self) -> Path:
		return Path(bpy.path.abspath(self.bl__log_file_path))

	####################
	# - Property Sync
	####################
	def sync_addon_logging(self, logger_to_setup: logging.Logger | None = None) -> None:
		"""Configure one, or all, active addon logger(s).

		Parameters:
			logger_to_setup:
				When set to None, all addon loggers will be configured
		"""
		if pydeps.DEPS_OK:
			log.info('Getting Logger (DEPS_OK = %s)', str(pydeps.DEPS_OK))
			with pydeps.importable_addon_deps(self.pydeps_path):
				from .utils import logger
		else:
			log.info('Getting Simple Logger (DEPS_OK = %s)', str(pydeps.DEPS_OK))
			logger = simple_logger

		# Retrieve Configured Log Levels
		log_level_console = logger.LOG_LEVEL_MAP[self.bl__log_level_console]
		log_level_file = logger.LOG_LEVEL_MAP[self.bl__log_level_file]

		log_setup_kwargs = {
			'console_level': log_level_console if self.use_log_console else None,
			'file_path': self.log_path if self.use_log_file else None,
			'file_level': log_level_file,
		}

		# Sync Single Logger / All Loggers
		if logger_to_setup is not None:
			logger.setup_logger(
				logger.console_handler,
				logger.file_handler,
				logger_to_setup,
				**log_setup_kwargs,
			)
		else:
			log.info('Re-Configuring All Loggers')
			logger.sync_all_loggers(
				logger.console_handler,
				logger.file_handler,
				**log_setup_kwargs,
			)

	def sync_use_default_pydeps_path(self, _: bpy.types.Context):
		# Switch to Default
		if self.use_default_pydeps_path:
			log.info(
				'Switching to Default PyDeps Path %s',
				str(info.DEFAULT_PATH_DEPS.resolve()),
			)
			self.cache__pydeps_path_while_using_default = self.bl__pydeps_path
			self.bl__pydeps_path = str(info.DEFAULT_PATH_DEPS.resolve())

		# Switch from Default
		else:
			log.info(
				'Switching from Default PyDeps Path %s to Cached PyDeps Path %s',
				str(info.DEFAULT_PATH_DEPS.resolve()),
				self.cache__pydeps_path_while_using_default,
			)
			self.bl__pydeps_path = self.cache__pydeps_path_while_using_default
			self.cache__pydeps_path_while_using_default = ''

	def sync_pydeps_path(self):
		if self.cache__backup_pydeps_path != self.bl__pydeps_path:
			log.info(
				'Syncing PyDeps Path from/to: %s => %s',
				self.cache__backup_pydeps_path,
				self.bl__pydeps_path,
			)
		else:
			log.info(
				'Syncing PyDeps Path In-Place @ %s',
				str(self.bl__pydeps_path),
			)

		# Error: Default Path in Use
		if self.use_default_pydeps_path:
			self.bl__pydeps_path = self.cache__backup_pydeps_path
			msg = "Can't update pydeps path while default path is being used"
			raise ValueError(msg)

		# Error: PyDeps Already Installed
		if pydeps.DEPS_OK:
			self.bl__pydeps_path = self.cache__backup_pydeps_path
			msg = "Can't update pydeps path while dependencies are installed"
			raise ValueError(msg)

		# Re-Check PyDeps
		log.info(
			'Checking PyDeps of New Path %s',
			str(self.pydeps_path),
		)
		if pydeps.check_pydeps(self.pydeps_path):
			# Re-Sync Loggers
			## We can now upgrade to the fancier loggers.
			self.sync_addon_logging()

			# Run Delayed Registrations
			## Since the deps are OK, we can now register the whole addon.
			registration.run_delayed_registration(
				registration.EVENT__DEPS_SATISFIED,
				self.pydeps_path,
			)

		# Backup New PyDeps Path
		self.cache__backup_pydeps_path = self.bl__pydeps_path

	####################
	# - UI
	####################
	def draw(self, _: bpy.types.Context) -> None:
		layout = self.layout
		num_pydeps_issues = len(pydeps.DEPS_ISSUES) if pydeps.DEPS_ISSUES else 0

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
		row.prop(self, 'bl__log_level_console')

		## Split Col: File Logging
		col = split.column()
		row = col.row()
		row.prop(self, 'use_log_file', toggle=True)

		row = col.row()
		row.enabled = self.use_log_file
		row.prop(self, 'bl__log_file_path')

		row = col.row()
		row.enabled = self.use_log_file
		row.prop(self, 'bl__log_level_file')

		# Box: Dependency Status
		box = layout.box()
		## Row: Header
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
		header.label(text=f'Install Mismatches ({num_pydeps_issues})')
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
		op.path_addon_pydeps = str(self.pydeps_path)
		op.path_addon_reqs = str(info.PATH_REQS)

		## Row: Uninstall
		row = box.row(align=True)
		op = row.operator(
			uninstall_deps.UninstallPyDeps.bl_idname,
			text='Uninstall PyDeps',
		)
		op.path_addon_pydeps = str(self.pydeps_path)


####################
# - Blender Registration
####################
BL_REGISTER = [
	BLMaxwellAddonPrefs,
]
