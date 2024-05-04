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

import logging
import shutil
import sys
import traceback
from pathlib import Path

import bpy

PATH_SCRIPT = str(Path(__file__).resolve().parent)
sys.path.insert(0, str(PATH_SCRIPT))
import info  # noqa: E402

sys.path.remove(str(PATH_SCRIPT))

# Set Bootstrap Log Level
## This will be the log-level of both console and file logs, at first...
## ...until the addon preferences have been loaded.
BOOTSTRAP_LOG_LEVEL = logging.DEBUG


def delete_addon_if_loaded(addon_name: str) -> bool:
	"""Strongly inspired by Blender's addon_utils.py."""
	removed_addon = False

	# Check if Python Module is Loaded
	mod = sys.modules.get(addon_name)
	# if (mod := sys.modules.get(addon_name)) is None:
	# ## It could still be loaded-by-default; then, it's in the prefs list
	# is_loaded_now = False
	# loads_by_default = addon_name in bpy.context.preferences.addons
	# else:
	# ## BL sets __addon_enabled__ on module of enabled addons.
	# ## BL sets __addon_persistent__ on module of load-by-default addons.
	# is_loaded_now = getattr(mod, '__addon_enabled__', False)
	# loads_by_default = getattr(mod, '__addon_persistent__', False)

	# Unregister Modules and Mark Disabled & Non-Persistent
	## This effectively disables it
	if mod is not None:
		removed_addon = True
		mod.__addon_enabled__ = False
		mod.__addon_persistent__ = False
		try:
			mod.unregister()
		except BaseException:
			traceback.print_exc()

	# Remove Addon
	## Remove Addon from Preferences
	## - Unsure why addon_utils has a while, but let's trust the process...
	while addon_name in bpy.context.preferences.addons:
		addon = bpy.context.preferences.addons.get(addon_name)
		if addon:
			bpy.context.preferences.addons.remove(addon)

	## Physically Excise Addon Code
	for addons_path in bpy.utils.script_paths(subdir='addons'):
		addon_path = Path(addons_path) / addon_name
		if addon_path.is_dir():
			shutil.rmtree(addon_path)

	## Save User Preferences
	bpy.ops.wm.save_userpref()

	return removed_addon


####################
# - Main
####################
if __name__ == '__main__':
	if delete_addon_if_loaded(info.ADDON_NAME):
		bpy.ops.wm.quit_blender()
		sys.exit(info.STATUS_UNINSTALLED_ADDON)
	else:
		bpy.ops.wm.quit_blender()
		sys.exit(info.STATUS_NOCHANGE_ADDON)
