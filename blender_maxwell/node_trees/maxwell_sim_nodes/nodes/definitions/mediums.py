from dataclasses import dataclass
import bpy
from .. import types, constants

####################
# - Utilities (to move out)
####################
def mk_set_preset(preset_dataclass):
	def set_preset(self, context):
		if self.preset in preset_dataclass.values:
			for parameter, value in preset_dataclass.values.items():
				self.inputs[parameter].default_value = value
	
	return set_preset


####################
# - Medium: Sellmeier
####################
@dataclass
class _Presets_SellmeierMaxwellMediumNode:
	descriptions = [
		('BK7', "BK7 Glass", "Borosilicate crown glass (known as BK7)"),
		('FUSED_SILICA', "Fused Silica", "Fused silica aka. SiO2"),
	]
	values = {
		"BK7": {
			"B1": 1.03961212,
			"B2": 0.231792344,
			"B3": 1.01046945,
			"C1 (um^2)": 6.00069867e-3,
			"C2 (um^2)": 2.00179144e-2,
			"C3 (um^2)": 103.560653,
		},
		"FUSED_SILICA": {
			"B1": 0.696166300,
			"B2": 0.407942600,
			"B3": 0.897479400,
			"C1 (um^2)": 4.67914826e-3,
			"C2 (um^2)": 1.35120631e-2,
			"C3 (um^2)": 97.9340025,
		},
	}
	
	def mk_set_preset(self):
		return mk_set_preset(self)

Presets_SellmeierMaxwellMediumNode = _Presets_SellmeierMaxwellMediumNode()

class SellmeierMaxwellMediumNode(types.MaxwellSimTreeNode, bpy.types.Node):
	bl_idname = types.SellmeierMaxwellMediumNodeType
	bl_label = "Sellmeier"
	bl_icon = constants.tree_constants.ICON_SIM_MEDIUM
	
	preset: bpy.props.EnumProperty(
		name="Presets",
		description="Select a preset",
		items=Presets_SellmeierMaxwellMediumNode.descriptions,
		default='BK7',
		update=Presets_SellmeierMaxwellMediumNode.mk_set_preset(),
	)

	def init(self, context):
		# Declare Node Inputs
		self.inputs.new('NodeSocketFloat', "B1")
		self.inputs.new('NodeSocketFloat', "B2")
		self.inputs.new('NodeSocketFloat', "B3")
		self.inputs.new('NodeSocketFloat', "C1 (um^2)")
		self.inputs.new('NodeSocketFloat', "C2 (um^2)")
		self.inputs.new('NodeSocketFloat', "C3 (um^2)")
		
		# Declare Node Outputs
		self.outputs.new(types.tree_types.MaxwellMediumSocketType, "Medium")
		
		# Set Preset Values
		Presets_SellmeierMaxwellMediumNode.mk_set_preset()(self, context)
	
	def draw_buttons(self, context, layout):
		layout.prop(self, 'sellmeier_presets', text="")



####################
# - Blender Registration
####################
BL_REGISTER = [
	SellmeierMaxwellMediumNode,
]
