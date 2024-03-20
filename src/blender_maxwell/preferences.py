import tomllib
from pathlib import Path

import bpy

from . import registration
from .operators_nodeps import install_deps, uninstall_deps
from .utils import logger as _logger
from .utils import pydeps

####################
# - Constants
####################
log = _logger.get()
PATH_ADDON_ROOT = Path(__file__).resolve().parent
with (PATH_ADDON_ROOT / 'pyproject.toml').open('rb') as f:
	PROJ_SPEC = tomllib.load(f)


####################
# - Preferences
####################
class BlenderMaxwellAddonPreferences(bpy.types.AddonPreferences):
	bl_idname = PROJ_SPEC['project']['name']  ## MUST match addon package name

	####################
	# - Properties
	####################
	# Default PyDeps Path
	use_default_path_addon_pydeps: bpy.props.BoolProperty(
		name='Use Default PyDeps Path',
		description='Whether to use the default PyDeps path',
		default=True,
		update=lambda self, context: self.sync_use_default_path_addon_pydeps(
			context
		),
	)
	cache_path_addon_pydeps: bpy.props.StringProperty(
		name='Cached Addon PyDeps Path',
		default=(_default_pydeps_path := str(pydeps.DEFAULT_PATH_DEPS)),
	)  ## Cache for use when toggling use of default pydeps path.
	## Must default to same as raw_path_* if default=True on use_default_*

	# Custom PyDeps Path
	raw_path_addon_pydeps: bpy.props.StringProperty(
		name='Addon PyDeps Path',
		description='Path to Addon Python Dependencies',
		subtype='FILE_PATH',
		default=_default_pydeps_path,
		update=lambda self, context: self.sync_path_addon_pydeps(context),
	)
	prev_raw_path_addon_pydeps: bpy.props.StringProperty(
		name='Previous Addon PyDeps Path',
		default=_default_pydeps_path,
	)  ## Use to restore raw_path_addon_pydeps after non-validated change.

	# TODO: LOGGING SETTINGS

	####################
	# - Property Sync
	####################
	def sync_use_default_path_addon_pydeps(self, _: bpy.types.Context):
		# Switch to Default
		if self.use_default_path_addon_pydeps:
			self.cache_path_addon_pydeps = self.raw_path_addon_pydeps
			self.raw_path_addon_pydeps = str(
				pydeps.DEFAULT_PATH_DEPS.resolve()
			)

		# Switch from Default
		else:
			self.raw_path_addon_pydeps = self.cache_path_addon_pydeps
			self.cache_path_addon_pydeps = ''

	def sync_path_addon_pydeps(self, _: bpy.types.Context):
		# Error if Default Path is in Use
		if self.use_default_path_addon_pydeps:
			self.raw_path_addon_pydeps = self.prev_raw_path_addon_pydeps
			msg = "Can't update pydeps path while default path is being used"
			raise ValueError(msg)

		# Error if Dependencies are All Installed
		if pydeps.DEPS_OK:
			self.raw_path_addon_pydeps = self.prev_raw_path_addon_pydeps
			msg = "Can't update pydeps path while dependencies are installed"
			raise ValueError(msg)

		# Update PyDeps
		## This also updates pydeps.DEPS_OK and pydeps.DEPS_ISSUES.
		## The result is used to run any delayed registrations...
		## ...which might be waiting for deps to be satisfied.
		if pydeps.check_pydeps(self.path_addon_pydeps):
			registration.run_delayed_registration(
				registration.EVENT__DEPS_SATISFIED,
				self.path_addon_pydeps,
			)
		self.prev_raw_path_addon_pydeps = self.raw_path_addon_pydeps

	####################
	# - Property Methods
	####################
	@property
	def path_addon_pydeps(self) -> Path:
		return Path(bpy.path.abspath(self.raw_path_addon_pydeps))

	@path_addon_pydeps.setter
	def path_addon_pydeps(self, value: Path) -> None:
		self.raw_path_addon_pydeps = str(value.resolve())

	####################
	# - UI
	####################
	def draw(self, _: bpy.types.Context) -> None:
		layout = self.layout
		num_pydeps_issues = (
			len(pydeps.DEPS_ISSUES) if pydeps.DEPS_ISSUES is not None else 0
		)

		# Box: Dependency Status
		box = layout.box()
		## Row: Header
		row = box.row(align=True)
		row.alignment = 'CENTER'
		row.label(text='Addon-Specific Python Deps')

		## Row: Toggle Default PyDeps Path
		row = box.row(align=True)
		row.enabled = not pydeps.DEPS_OK
		row.prop(
			self,
			'use_default_path_addon_pydeps',
			text='Use Default PyDeps Install Path',
			toggle=True,
		)

		## Row: Current PyDeps Path
		row = box.row(align=True)
		row.enabled = (
			not pydeps.DEPS_OK and not self.use_default_path_addon_pydeps
		)
		row.prop(self, 'raw_path_addon_pydeps', text='PyDeps Install Path')

		## Row: More Information Panel
		row = box.row(align=True)
		header, panel = row.panel('pydeps_issues', default_closed=True)
		header.label(text=f'Dependency Conflicts ({num_pydeps_issues})')
		if panel is not None:
			grid = panel.grid_flow()
			for issue in pydeps.DEPS_ISSUES:
				grid.label(text=issue)

		## Row: Install
		row = box.row(align=True)
		row.enabled = not pydeps.DEPS_OK
		op = row.operator(
			install_deps.InstallPyDeps.bl_idname,
			text='Install PyDeps',
		)
		op.path_addon_pydeps = str(self.path_addon_pydeps)
		op.path_addon_reqs = str(pydeps.PATH_REQS)

		## Row: Uninstall
		row = box.row(align=True)
		row.enabled = pydeps.DEPS_OK
		op = row.operator(
			uninstall_deps.UninstallPyDeps.bl_idname,
			text='Uninstall PyDeps',
		)
		op.path_addon_pydeps = str(self.path_addon_pydeps)


####################
# - Blender Registration
####################
BL_REGISTER = [
	BlenderMaxwellAddonPreferences,
]
