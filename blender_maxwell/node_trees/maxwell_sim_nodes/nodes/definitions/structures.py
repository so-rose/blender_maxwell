import bpy
from .. import types, constants

class TriMeshMaxwellStructureNode(types.MaxwellSimTreeNode, bpy.types.Node):
	bl_idname = types.TriMeshMaxwellStructureNodeType
	bl_label = "TriMesh"
	bl_icon = constants.tree_constants.ICON_SIM_STRUCTURE

	# Initialization - Called on Node Creation
	def init(self, context):
		# Declare Node Inputs
		self.inputs.new('NodeSocketObject', "Object")
		self.inputs.new(types.tree_types.MaxwellMediumSocketType, "Medium")
		
		# Declare Node Outputs
		self.outputs.new(types.tree_types.MaxwellStructureSocketType, "Structure")



####################
# - Blender Registration
####################
BL_REGISTER = [
	TriMeshMaxwellStructureNode,
]
