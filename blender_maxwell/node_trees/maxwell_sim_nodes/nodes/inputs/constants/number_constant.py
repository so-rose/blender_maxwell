import bpy
import sympy as sp

from .... import contracts
from .... import sockets
from ... import base

class NumberConstantNode(base.MaxwellSimTreeNode):
	node_type = contracts.NodeType.NumberConstant
	
	bl_label = "Numerical Constant"
	#bl_icon = constants.ICON_SIM_INPUT
	
	input_sockets = {
		"value": sockets.ComplexNumberSocketDef(
			label="Complex",
		),  ## TODO: Dynamic number socket!
	}
	output_sockets = {
		"value": sockets.ComplexNumberSocketDef(
			label="Complex",
		),  ## TODO: Dynamic number socket!
	}
	
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
