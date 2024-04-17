import subprocess
import sys
from pathlib import Path

import bpy

from blender_maxwell.utils import logger as _logger

from .. import registration

log = _logger.get(__name__)


class InstallPyDeps(bpy.types.Operator):
	bl_idname = 'blender_maxwell.nodeps__addon_install_popup'
	bl_label = 'Popup to Install BLMaxwell Python Deps'

	path_addon_pydeps: bpy.props.StringProperty(
		name='Path to Addon Python Dependencies',
		default='',
	)
	path_addon_reqs: bpy.props.StringProperty(
		name='Path to Addon Python Dependencies',
		default='',
	)

	# TODO: poll()

	def execute(self, _: bpy.types.Context):
		if self.path_addon_pydeps == '' or self.path_addon_reqs == '':
			msg = f"A path for operator {self.bl_idname} isn't set"
			raise ValueError(msg)

		path_addon_pydeps = Path(self.path_addon_pydeps)
		path_addon_reqs = Path(self.path_addon_reqs)
		log.info(
			'Running Install PyDeps w/requirements.txt (%s) to path: %s',
			path_addon_reqs,
			path_addon_pydeps,
		)

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
			cmdline = [
				str(python_exec),
				'-m',
				'pip',
				'install',
				'-r',
				str(path_addon_reqs),
				'--target',
				str(path_addon_pydeps),
			]
			log.info(
				'Running pip w/cmdline: %s',
				' '.join(cmdline),
			)
			subprocess.check_call(cmdline)
		except subprocess.CalledProcessError:
			log.exception('Failed to install PyDeps')
			return {'CANCELLED'}

		registration.run_delayed_registration(
			registration.EVENT__DEPS_SATISFIED,
			path_addon_pydeps,
		)
		return {'FINISHED'}


####################
# - Blender Registration
####################
BL_REGISTER = [
	InstallPyDeps,
]
BL_KEYMAP_ITEM_DEFS = []
