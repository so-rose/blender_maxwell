import zipfile
import contextlib
import shutil
import sys
from pathlib import Path

import bpy
import addon_utils

PATH_ROOT = Path(__file__).resolve().parent

####################
# - Defined Constants
####################
ADDON_NAME = "blender_maxwell"
PATH_BLEND = PATH_ROOT / "demo.blend"
PATH_ADDON_DEPS = PATH_ROOT / ".cached-dependencies"

####################
# - Computed Constants
####################
PATH_ADDON = PATH_ROOT / ADDON_NAME
PATH_ADDON_ZIP = PATH_ROOT / (ADDON_NAME + ".zip")

####################
# - Utilities
####################
@contextlib.contextmanager
def zipped_directory(path_dir: Path, path_zip: Path):
	"""Context manager that exposes a zipped version of a directory,
	then deletes the .zip file afterwards.
	"""
	# Delete Existing ZIP file (if exists)
	if path_zip.is_file(): path_zip.unlink()
	
	# Create a (new) ZIP file of the addon directory
	with zipfile.ZipFile(path_zip, 'w', zipfile.ZIP_DEFLATED) as f_zip:
		for file_to_zip in path_dir.rglob('*'):
			f_zip.write(file_to_zip, file_to_zip.relative_to(path_dir.parent))
	
	# Delete the ZIP
	try:
		yield path_zip
	finally:
		path_zip.unlink()

####################
# - main()
####################
if __name__ == "__main__":
	# Check and uninstall the addon if it's enabled
	is_loaded_by_default, is_loaded_now = addon_utils.check(ADDON_NAME)
	if is_loaded_now:
		# Disable the Addon
		addon_utils.disable(ADDON_NAME, default_set=True, handle_error=None)

		# Completey Delete the Addon
		for mod in addon_utils.modules():
			if mod.__name__ == ADDON_NAME:
				# Delete Addon from Blender Python Tree
				shutil.rmtree(Path(mod.__file__).parent)
				
				# Reset All Addons
				addon_utils.reset_all()
				
				# Save User Preferences & Break
				bpy.ops.wm.save_userpref()
				break
		
		# Quit Blender (hard-flush Python environment)
		## - Python environments are not made to be partially flushed.
		## - This is the only truly reliable way to avoid all bugs.
		## - See https://github.com/JacquesLucke/blender_vscode
		bpy.ops.wm.quit_blender()
		try:
			raise RuntimeError
		except:
			sys.exit(42)

	with zipped_directory(PATH_ADDON, PATH_ADDON_ZIP) as path_zipped:
		# Install the ZIPped Addon
		bpy.ops.preferences.addon_install(filepath=str(path_zipped))
	
	# Enable the Addon
	addon_utils.enable(
		ADDON_NAME,
		default_set=True,
		persistent=True,
		handle_error=None,
	)
	
	# Save User Preferences
	bpy.ops.wm.save_userpref()

	# Load the .blend
	bpy.ops.wm.open_mainfile(filepath=str(PATH_BLEND))
	
	# Ensure Addon-Specific Dependency Cache is Importable
	## - In distribution, the addon keeps this folder in the Blender script tree.
	## - For testing, we need to hack sys.path here.
	## - This avoids having to install all deps with every reload.
	if str(PATH_ADDON_DEPS) not in sys.path:
		sys.path.insert(0, str(PATH_ADDON_DEPS))

	# Modify any specific settings, if needed
	# Example: bpy.context.preferences.addons[addon_name].preferences.your_setting = "your_value"


