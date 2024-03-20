"""Blender startup script ensuring correct addon installation.

See <https://github.com/dfelinto/blender/blob/master/release/scripts/modules/addon_utils.py>
"""

import shutil
import sys
import traceback
from pathlib import Path

import bpy

sys.path.insert(0, str(Path(__file__).resolve().parent))
import info
import pack

## TODO: Preferences item that allows using BLMaxwell 'starter.blend' as Blender's default starter blendfile.


####################
# - Addon Functions
####################
def delete_addon_if_loaded(addon_name: str) -> None:
	"""Strongly inspired by Blender's addon_utils.py."""
	should_restart_blender = False

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
		mod.__addon_enabled__ = False
		mod.__addon_persistent__ = False
		try:
			mod.unregister()
		except BaseException:
			traceback.print_exc()
			should_restart_blender = True

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
		if addon_path.exists():
			shutil.rmtree(addon_path)
			should_restart_blender = True

	## Save User Preferences
	bpy.ops.wm.save_userpref()

	# Quit (Restart) Blender - hard-flush Python environment
	## - Python environments are not made to be partially flushed.
	## - This is the only truly reliable way to avoid all bugs.
	## - See <https://github.com/JacquesLucke/blender_vscode>
	## - By passing STATUS_UNINSTALLED_ADDON, we report that it's clean now.
	if should_restart_blender:
		bpy.ops.wm.quit_blender()
		sys.exit(info.STATUS_UNINSTALLED_ADDON)


def install_addon(addon_name: str, addon_zip: Path) -> None:
	"""Strongly inspired by Blender's addon_utils.py."""
	# Check if Addon is Installable
	if any(
		[
			(mod := sys.modules.get(addon_name)) is not None,
			addon_name in bpy.context.preferences.addons,
			any(
				(Path(addon_path) / addon_name).exists()
				for addon_path in bpy.utils.script_paths(subdir='addons')
			),
		]
	):
		## TODO: Check if addon file path exists?
		in_pref_addons = addon_name in bpy.context.preferences.addons
		existing_files_found = {
			addon_path: (Path(addon_path) / addon_name).exists()
			for addon_path in bpy.utils.script_paths(subdir='addons')
			if (Path(addon_path) / addon_name).exists()
		}
		msg = f"Addon (module = '{mod}') is not installable (in preferences.addons: {in_pref_addons}) (existing files found: {existing_files_found})"
		raise ValueError(msg)

	# Install Addon
	bpy.ops.preferences.addon_install(filepath=str(addon_zip))
	if not any(
		(Path(addon_path) / addon_name).exists()
		for addon_path in bpy.utils.script_paths(subdir='addons')
	):
		msg = f"Couldn't install addon {addon_name}"
		raise RuntimeError(msg)

	# Enable Addon
	bpy.ops.preferences.addon_enable(module=addon_name)
	if addon_name not in bpy.context.preferences.addons:
		msg = f"Couldn't enable addon {addon_name}"
		raise RuntimeError(msg)

	# Set Dev Path for Addon Dependencies
	addon_prefs = bpy.context.preferences.addons[addon_name].preferences
	addon_prefs.use_default_path_addon_pydeps = False
	addon_prefs.path_addon_pydeps = info.PATH_ADDON_DEV_DEPS

	# Save User Preferences
	bpy.ops.wm.save_userpref()


####################
# - Entrypoint
####################
if __name__ == '__main__':
	# Delete Addon (maybe; possibly restart)
	delete_addon_if_loaded(info.ADDON_NAME)

	# Signal that Live-Printing can Start
	print(info.SIGNAL_START_CLEAN_BLENDER)  # noqa: T201

	# Install and Enable Addon
	install_failed = False
	with pack.zipped_addon(
		info.PATH_ADDON_PKG,
		info.PATH_ADDON_ZIP,
		info.PATH_ROOT / 'pyproject.toml',
		info.PATH_ROOT / 'requirements.lock',
	) as path_zipped:
		try:
			install_addon(info.ADDON_NAME, path_zipped)
		except Exception as exe:
			traceback.print_exc()
			install_failed = True

	# Load Development .blend
	## TODO: We need a better (also final-deployed-compatible) solution for what happens when a user opened a .blend file without installing dependencies!
	if not install_failed:
		bpy.ops.wm.open_mainfile(filepath=str(info.PATH_ADDON_DEV_BLEND))
	else:
		bpy.ops.wm.quit_blender()
		sys.exit(info.STATUS_NOINSTALL_ADDON)
