# blender_maxwell
# Copyright (C) 2024 blender_maxwell Project Contributors
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import shutil
import site
from pathlib import Path

import bpy

from blender_maxwell import contracts as ct

from ..utils import pip_process, pydeps, simple_logger

log = simple_logger.get(__name__)


class UninstallPyDeps(bpy.types.Operator):
	bl_idname = ct.OperatorType.UninstallPyDeps
	bl_label = 'Uninstall BLMaxwell Python Deps'

	@classmethod
	def poll(cls, _: bpy.types.Context):
		return not pip_process.is_loaded() and (
			pydeps.DEPS_OK or (pydeps.DEPS_ISSUES and pydeps.DEPS_INST_DEPLOCKS)
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
	# - Execution
	####################
	def execute(self, _: bpy.types.Context):
		# Reject Bad PyDeps Paths (to prevent unfortunate deletions)
		## Reject user site-packages
		if self.pydeps_path == Path(site.getusersitepackages()):
			msg = f"PyDeps path ({self.pydeps_path}) can't be the user site-packages"
			raise ValueError(msg)

		## Reject any global site-packages
		if self.pydeps_path == Path(site.getusersitepackages()):
			msg = f"PyDeps path ({self.pydeps_path}) can't be a global site-packages"
			raise ValueError(msg)

		## Reject any Reserved sys.path Entry (as of addon initialization)
		## -> At addon init, ORIGINAL_SYS_PATH is created as a sys.path copy.
		## -> Thus, ORIGINAL_SYS_PATH only includes Blender-set paths.
		## -> (possibly also other addon's manipulations, but that's good!)
		if self.pydeps_path in [
			Path(sys_path) for sys_path in ct.addon.ORIGINAL_SYS_PATH
		]:
			msg = f'PyDeps path ({self.pydeps_path}) can\'t be any package defined in "sys.path"'
			raise ValueError(msg)

		## Reject non-existant PyDeps Path
		if not self.pydeps_path.exists():
			msg = f"PyDeps path ({self.pydeps_path}) doesn't exist"
			raise ValueError(msg)

		## Reject non-directory PyDeps Path
		if not self.pydeps_path.is_dir():
			msg = f"PyDeps path ({self.pydeps_path}) isn't a directory"
			raise ValueError(msg)

		## Reject PyDeps Path that is Home Dir (I hope nobody needs this)
		if self.pydeps_path == Path.home().resolve():
			msg = f"PyDeps path ({self.pydeps_path}) can't be the user home directory"
			raise ValueError(msg)

		# Check for Empty Directory
		if len(pydeps.compute_installed_deplocks(self.pydeps_path)) == 0:
			## Reject Non-Empty Directories w/o Python Dependencies
			if any(Path(self.pydeps_path).iterdir()):
				msg = "PyDeps Path has no installed Python modules, but isn't empty: {self.pydeps_path)"
				raise ValueError(msg)

			self.report(
				{'ERROR'},
				f"PyDeps Path is empty; uninstall can't run: {self.pydeps_path}",
			)
			return {'FINISHED'}

		# Brutally Delete / Remake PyDeps Folder
		## The point isn't to protect against dedicated stupididy.
		## Just to nudge away a few of the obvious "bad ideas" users might have.
		## TODO: Handle rmtree.avoids_symlink_attacks
		## TODO: Handle audit events
		log.warning(
			'Deleting and Creating Folder at "%s": %s',
			'pydeps_path',
			str(self.pydeps_path),
		)
		shutil.rmtree(self.pydeps_path)
		self.pydeps_path.mkdir()

		# Update Changed PyDeps
		ct.addon.prefs().on_addon_pydeps_changed()
		return {'FINISHED'}


####################
# - Blender Registration
####################
BL_REGISTER = [
	UninstallPyDeps,
]
BL_HOTKEYS = []
