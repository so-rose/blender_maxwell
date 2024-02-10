import bpy

from . import types, constants

class MaxwellSimTree(bpy.types.NodeTree):
	bl_idname = types.TreeType.MaxwellSim
	bl_label = "Maxwell Sim Editor"
	bl_icon = constants.ICON_SIM	## Icon ID



####################
# - Blender Registration
####################
BL_REGISTER = [
	MaxwellSimTree,
]
