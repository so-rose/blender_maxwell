import bpy
from .... import types, constants
from ... import node_base

class TripleSellmeierMediumNode(node_base.MaxwellSimTreeNode):
	bl_idname = types.NodeType.TripleSellmeierMedium.value
	bl_label = "Triple Sellmeier Medium"
	bl_icon = constants.ICON_SIM_MEDIUM

	input_sockets = {
		"B1": ("NodeSocketFloat", "B1"),
		"B2": ("NodeSocketFloat", "B2"),
		"B3": ("NodeSocketFloat", "B3"),
		"C1": ("NodeSocketFloat", "C1 (um^2)"),
		"C2": ("NodeSocketFloat", "C2 (um^2)"),
		"C3": ("NodeSocketFloat", "C3 (um^2)"),
	}
	output_sockets = {
		"medium": (types.SocketType.MaxwellMedium, "Medium")
	}
	socket_presets = {
		"_description": [
			('BK7', "BK7 Glass", "Borosilicate crown glass (known as BK7)"),
			('FUSED_SILICA', "Fused Silica", "Fused silica aka. SiO2"),
		],
		"_default": "BK7",
		"_values": {
			"BK7": {
				"B1": 1.03961212,
				"B2": 0.231792344,
				"B3": 1.01046945,
				"C1": 6.00069867e-3,
				"C2": 2.00179144e-2,
				"C3": 103.560653,
			},
			"FUSED_SILICA": {
				"B1": 0.696166300,
				"B2": 0.407942600,
				"B3": 0.897479400,
				"C1": 4.67914826e-3,
				"C2": 1.35120631e-2,
				"C3": 97.9340025,
			},
		}
	}
	
	####################
	# - Properties
	####################
	def draw_buttons(self, context, layout):
		layout.prop(self, 'preset', text="")



####################
# - Blender Registration
####################
BL_REGISTER = [
	TripleSellmeierMediumNode,
]
BL_NODES = {
	types.NodeType.TripleSellmeierMedium: (
		types.NodeCategory.MAXWELL_SIM_MEDIUMS_LINEARMEDIUMS
	)
}
