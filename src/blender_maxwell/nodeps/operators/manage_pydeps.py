from pathlib import Path

import bpy

from blender_maxwell import contracts as ct

from ..utils import pip_process, pydeps, simple_logger

log = simple_logger.get(__name__)


class ManagePyDeps(bpy.types.Operator):
	bl_idname = ct.OperatorType.ManagePyDeps
	bl_label = 'Blender Maxwell Python Dependency Manager'
	bl_options = {'REGISTER'}

	show_pydeps_conflicts: bpy.props.BoolProperty(
		name='Show Conflicts',
		description='Show the conflicts between installed and required packages.',
		default=False,
	)
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
	# - UI
	####################
	def draw(self, _: bpy.types.Context) -> None:
		layout = self.layout

		## Row: Toggle Default PyDeps Path
		row = layout.row()
		row.alignment = 'CENTER'
		row.label(
			text="Blender Maxwell relies on Python dependencies that aren't currently satisfied."
		)
		row.prop(
			self,
			'show_pydeps_conflicts',
			text=f'Show Conflicts ({len(pydeps.DEPS_ISSUES)})',
			toggle=True,
		)

		## Grid: Issues Panel
		if self.show_pydeps_conflicts:
			grid = layout.grid_flow()
			grid.alignment = 'CENTER'
			for issue in pydeps.DEPS_ISSUES:
				grid.label(text=issue)

		# Row: Install Deps
		row = layout.row(align=True)
		op = row.operator(
			ct.OperatorType.InstallPyDeps,
			text='Install Python Dependencies (requires internet)',
		)
		op.bl__pydeps_path = str(self.pydeps_path)

		## Row: Uninstall Deps
		row = layout.row(align=True)
		op = row.operator(
			ct.OperatorType.UninstallPyDeps,
			text='Uninstall Python Dependencies',
		)
		op.bl__pydeps_path = str(self.pydeps_path)

		## Row: Deps Install Progress
		row = layout.row()
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

		## Row: Toggle Default PyDeps Path
		row = layout.row()
		row.alignment = 'CENTER'
		row.label(
			text='After installation, the addon is ready to use. For more details, please refer to the addon preferences.'
		)

	####################
	# - Execute
	####################
	def invoke(self, context: bpy.types.Context, event: bpy.types.Event):
		if not bpy.app.background:
			# Force-Move Mouse Cursor to Window Center
			## This forces the popup dialog to spawn in the center of the screen.
			context.window.cursor_warp(
				context.window.width // 2,
				context.window.height // 2 + 2 * bpy.context.preferences.system.dpi,
			)

			# Spawn Popup Dialogue
			return context.window_manager.invoke_props_dialog(
				self, width=8 * bpy.context.preferences.system.dpi
			)

		log.info('Skipping ManagePyDeps popup, since Blender is running without a GUI')
		return {'INTERFACE'}

	def execute(self, _: bpy.types.Context):
		if not pydeps.DEPS_OK:
			self.report(
				{'ERROR'},
				f'Python Dependencies for "{ct.addon.NAME}" were not installed. Please refer to the addon preferences.',
			)
		return {'FINISHED'}


####################
# - Blender Registration
####################
BL_REGISTER = [
	ManagePyDeps,
]
BL_HOTKEYS = []
