import bpy
from .. import types, constants

import tidy3d as td

class PointDipoleMaxwellSourceNode(types.MaxwellSimTreeNode, bpy.types.Node):
	bl_idname = types.PointDipoleMaxwellSourceNodeType
	bl_label = "Point Dipole"
	bl_icon = constants.tree_constants.ICON_SIM_SOURCE

	input_sockets = {
		"center": ("NodeSocketVector", "Center"),
		"interpolate": ("NodeSocketBool", "Interpolate"),
	}
	output_sockets = {
		"source": (types.tree_types.MaxwellSourceSocketType, "Source")
	}
	
	####################
	# - Properties
	####################
	polarization: bpy.props.EnumProperty(
		name="Polarization",
		description="Polarization of the generated point dipole field",
		items=[
			("Ex", "Ex", "x-component of E-field"),
			("Ey", "Ey", "y-component of E-field"),
			("Ez", "Ez", "z-component of E-field"),
			("Hx", "Hx", "x-component of H-field"),
			("Hy", "Hy", "y-component of H-field"),
			("Hz", "Hz", "z-component of H-field"),
		],
		default="Ex",
	)
	
	####################
	# - Node UI and Layout
	####################
	def draw_buttons(self, context, layout):
		layout.prop(self, 'polarization', text="")
		
	####################
	# - Socket Properties
	####################
	@types.output_socket_cb("source")
	def output_source(self):
		return td.PointDipole(
			center=tuple(self.compute_input("center")),
			size=(0, 0, 0),
			source_time=td.GaussianPulse(freq0=200e12, fwidth=200e12),
			## ^ Placeholder
			interpolate=self.compute_input("interpolate"),
			polarization=str(self.polarization),
		)



####################
# - Blender Registration
####################
BL_REGISTER = [
	PointDipoleMaxwellSourceNode,
]
