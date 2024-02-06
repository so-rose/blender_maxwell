import sys
import shutil
import subprocess
from pathlib import Path

import bpy

from . import types

class BlenderMaxwellUninstallDependenciesOperator(bpy.types.Operator):
	bl_idname = types.BlenderMaxwellUninstallDependencies
	bl_label = "Uninstall Dependencies for Blender Maxwell Addon"

	def execute(self, context):
		addon_dir = Path(__file__).parent.parent
		#addon_specific_folder = addon_dir / '.dependencies'
		addon_specific_folder = Path("/home/sofus/src/college/bsc_ge/thesis/code/.cached-dependencies")
		
		shutil.rmtree(addon_specific_folder)
		
		return {'FINISHED'}



####################
# - Blender Registration
####################
BL_REGISTER = [
	BlenderMaxwellUninstallDependenciesOperator,
]
