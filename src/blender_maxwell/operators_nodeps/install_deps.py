import subprocess
import sys
from pathlib import Path

import bpy

from .. import registration


class InstallPyDeps(bpy.types.Operator):
	bl_idname = 'blender_maxwell.nodeps__install_py_deps'
	bl_label = 'Install BLMaxwell Python Deps'

	path_addon_pydeps: bpy.props.StringProperty(
		name='Path to Addon Python Dependencies'
	)
	path_addon_reqs: bpy.props.StringProperty(
		name='Path to Addon Python Dependencies'
	)

	def execute(self, _: bpy.types.Context):
		path_addon_pydeps = Path(self.path_addon_pydeps)
		path_addon_reqs = Path(self.path_addon_reqs)

		# Create the Addon-Specific Folder (if Needed)
		## It MUST, however, have a parent already
		path_addon_pydeps.mkdir(parents=False, exist_ok=True)

		# Determine Path to Blender's Bundled Python
		## bpy.app.binary_path_python was deprecated in 2.91.
		## sys.executable points to the correct bundled Python.
		## See <https://developer.blender.org/docs/release_notes/2.91/python_api/>
		python_exec = Path(sys.executable)

		# Install Deps w/Bundled pip
		try:
			subprocess.check_call(
				[
					str(python_exec),
					'-m',
					'pip',
					'install',
					'-r',
					str(path_addon_reqs),
					'--target',
					str(path_addon_pydeps),
				]
			)
		except subprocess.CalledProcessError as e:
			msg = f'Failed to install dependencies: {str(e)}'
			self.report({'ERROR'}, msg)
			return {'CANCELLED'}

		registration.run_delayed_registration(
			registration.EVENT__ON_DEPS_INSTALLED,
			path_addon_pydeps,
		)
		return {'FINISHED'}


####################
# - Blender Registration
####################
BL_REGISTER = [
	InstallPyDeps,
]
