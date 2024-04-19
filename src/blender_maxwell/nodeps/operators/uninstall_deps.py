import shutil
from pathlib import Path

import bpy

from blender_maxwell import contracts as ct

from ..utils import pydeps


class UninstallPyDeps(bpy.types.Operator):
	bl_idname = ct.OperatorType.UninstallPyDeps
	bl_label = 'Uninstall BLMaxwell Python Deps'

	@classmethod
	def poll(cls, _: bpy.types.Context):
		return pydeps.DEPS_OK

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
	# - Execution
	####################
	def execute(self, _: bpy.types.Context):
		path_addon_pydeps = Path(self.pydeps_path)
		if (
			pydeps.check_pydeps()
			and path_addon_pydeps.exists()
			and path_addon_pydeps.is_dir()
		):
			raise NotImplementedError
			# TODO: CAREFUL!!
			# shutil.rmtree(self.path_addon_pydeps)
		else:
			msg = "Can't uninstall pydeps"
			raise RuntimeError(msg)

		return {'FINISHED'}


####################
# - Blender Registration
####################
BL_REGISTER = [
	UninstallPyDeps,
]
BL_HOTKEYS = []
