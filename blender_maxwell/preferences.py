import bpy

from .operators import types as operators_types

class BlenderMaxwellAddonPreferences(bpy.types.AddonPreferences):
    bl_idname = "blender_maxwell"

    def draw(self, context):
        layout = self.layout
        layout.operator(operators_types.BlenderMaxwellInstallDependencies, text="Install Dependencies")
        layout.operator(operators_types.BlenderMaxwellUninstallDependencies, text="Uninstall Dependencies")

####################
# - Blender Registration
####################
BL_REGISTER = [
	BlenderMaxwellAddonPreferences,
]
