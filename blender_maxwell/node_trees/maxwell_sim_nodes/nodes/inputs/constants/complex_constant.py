import bpy
import sympy as sp

from .... import contracts
from .... import sockets
from ... import base

class ComplexConstantNode(base.MaxwellSimTreeNode):
	node_type = contracts.NodeType.ComplexConstant
	
	bl_label = "Complex Constant"
	#bl_icon = constants.ICON_SIM_INPUT
	
	input_sockets = {
		"value": sockets.ComplexNumberSocketDef(
			label="Complex",
		),
	}
	output_sockets = {
		"value": sockets.ComplexNumberSocketDef(
			label="Complex",
		),
	}
	
	####################
	# - Callbacks
	####################
	@base.computes_output_socket("value", sp.Expr)
	def compute_value(self: contracts.NodeTypeProtocol) -> sp.Expr:
		return self.compute_input("value")



####################
# - Blender Registration
####################
BL_REGISTER = [
	ComplexConstantNode,
]
BL_NODES = {
	contracts.NodeType.ComplexConstant: (
		contracts.NodeCategory.MAXWELL_SIM_INPUTS_CONSTANTS
	)
}
