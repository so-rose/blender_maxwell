import sys
from pathlib import Path

import bpy

PATH_SCRIPT = str(Path(__file__).resolve().parent)
sys.path.insert(0, str(PATH_SCRIPT))
import info  # noqa: E402
import pack  # noqa: E402

sys.path.remove(str(PATH_SCRIPT))


def install_and_enable_addon(addon_name: str, addon_zip: Path) -> None:
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

	# Save User Preferences
	bpy.ops.wm.save_userpref()


def setup_for_development(addon_name: str, path_addon_dev_deps: Path) -> None:
	addon_prefs = bpy.context.preferences.addons[addon_name].preferences

	# PyDeps Path
	addon_prefs.use_default_pydeps_path = False
	addon_prefs.pydeps_path = path_addon_dev_deps

	# Save User Preferences
	bpy.ops.wm.save_userpref()


####################
# - Main
####################
if __name__ == '__main__':
	with pack.zipped_addon(
		info.PATH_ADDON_PKG,
		info.PATH_ADDON_ZIP,
		info.PATH_ROOT / 'pyproject.toml',
		info.PATH_ROOT / 'requirements.lock',
		initial_log_level=info.BOOTSTRAP_LOG_LEVEL,
	) as path_zipped:
		install_and_enable_addon(info.ADDON_NAME, path_zipped)

	setup_for_development(info.ADDON_NAME, info.PATH_ADDON_DEV_DEPS)

	bpy.ops.wm.quit_blender()
	sys.exit(info.STATUS_INSTALLED_ADDON)
