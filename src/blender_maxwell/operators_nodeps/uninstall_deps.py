import shutil

import bpy

from ..utils import pydeps
from .. import registration


class UninstallPyDeps(bpy.types.Operator):
	bl_idname = 'blender_maxwell.nodeps__uninstall_py_deps'
	bl_label = 'Uninstall BLMaxwell Python Deps'

	path_addon_pydeps: bpy.props.StringProperty(
		name='Path to Addon Python Dependencies'
	)

	def execute(self, _: bpy.types.Context):
		if (
			pydeps.check_pydeps()
			and self.path_addon_pydeps.exists()
			and self.path_addon_pydeps.is_dir()
		):
			# CAREFUL!!
			shutil.rmtree(self.path_addon_pydeps)
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
