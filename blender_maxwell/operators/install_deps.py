import sys
import subprocess
from pathlib import Path

import bpy

from . import types

class BlenderMaxwellInstallDependenciesOperator(bpy.types.Operator):
	bl_idname = types.BlenderMaxwellInstallDependencies
	bl_label = "Install Dependencies for Blender Maxwell Addon"

	def execute(self, context):
		addon_dir = Path(__file__).parent.parent
		requirements_path = addon_dir / 'requirements.txt'
		#addon_specific_folder = addon_dir / '.dependencies'
		addon_specific_folder = Path("/home/sofus/src/college/bsc_ge/thesis/code/.cached-dependencies")
		
		# Create the Addon-Specific Folder
		addon_specific_folder.mkdir(parents=True, exist_ok=True)
		
		# Determine Path to Blender's Bundled Python
		python_exec = Path(sys.executable)
		## bpy.app.binary_path_python was deprecated in 2.91.
		## sys.executable points to the correct bundled Python.
		## See <https://developer.blender.org/docs/release_notes/2.91/python_api/>
		
		# Install Dependencies w/Bundled pip
		try:
			subprocess.check_call([
				str(python_exec), '-m',
				'pip', 'install',
				'-r', str(requirements_path), 
				'--target', str(addon_specific_folder),
			])
			self.report(
				{'INFO'},
				"Dependencies for 'blender_maxwell' installed successfully."
			)
		except subprocess.CalledProcessError as e:
			self.report(
				{'ERROR'},
				f"Failed to install dependencies: {str(e)}"
			)
			return {'CANCELLED'}
		
		# Install Dependencies w/Bundled pip
		if str(addon_specific_folder) not in sys.path:
			sys.path.insert(0, str(addon_specific_folder))
		
		return {'FINISHED'}



####################
# - Blender Registration
####################
BL_REGISTER = [
	BlenderMaxwellInstallDependenciesOperator,
]
