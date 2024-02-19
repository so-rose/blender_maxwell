import bpy
import sympy as sp

from .... import contracts
from .... import sockets
from ... import base

class PhysicalConstantNode(base.MaxwellSimTreeNode):
	node_type = contracts.NodeType.PhysicalConstant
	
	bl_label = "Physical Constant"
	#bl_icon = constants.ICON_SIM_INPUT
	
	input_sockets = {}
	input_socket_sets = {
		"time": {
			"value": sockets.PhysicalTimeSocketDef(
				label="Time",
			),
		},
		"angle": {
			"value": sockets.PhysicalAngleSocketDef(
				label="Angle",
			),
		},
		"length": {
			"value": sockets.PhysicalLengthSocketDef(
				label="Length",
			),
		},
		"area": {
			"value": sockets.PhysicalAreaSocketDef(
				label="Area",
			),
		},
		"volume": {
			"value": sockets.PhysicalVolumeSocketDef(
				label="Volume",
			),
		},
		"point_3d": {
			"value": sockets.PhysicalPoint3DSocketDef(
				label="3D Point",
			),
		},
		"size_3d": {
			"value": sockets.PhysicalSize3DSocketDef(
				label="3D Size",
			),
		},
		## I got bored so maybe the rest later
	}
	output_sockets = {}
	output_socket_sets = input_socket_sets
	
	####################
	# - Callbacks
	####################
	@base.computes_output_socket("value")
	def compute_value(self: contracts.NodeTypeProtocol) -> sp.Expr:
		return self.compute_input("value")



####################
# - Blender Registration
####################
BL_REGISTER = [
	PhysicalConstantNode,
]
BL_NODES = {
	contracts.NodeType.PhysicalConstant: (
		contracts.NodeCategory.MAXWELLSIM_INPUTS_CONSTANTS
	)
}
