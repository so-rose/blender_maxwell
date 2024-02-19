import bpy
import sympy as sp

from .... import contracts
from .... import sockets
from ... import base

class NumberConstantNode(base.MaxwellSimTreeNode):
	node_type = contracts.NodeType.NumberConstant
	
	bl_label = "Numerical Constant"
	#bl_icon = constants.ICON_SIM_INPUT
	
	input_sockets = {}
	input_socket_sets = {
		"real": {
			"value": sockets.RealNumberSocketDef(
				label="Real",
			),
		},
		"complex": {
			"value": sockets.ComplexNumberSocketDef(
				label="Complex",
			),
		},
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
	NumberConstantNode,
]
BL_NODES = {
	contracts.NodeType.NumberConstant: (
		contracts.NodeCategory.MAXWELLSIM_INPUTS_CONSTANTS
	)
}
