import typing as typ

import bpy
import sympy as sp

from .... import contracts as ct
from .... import sockets
from ... import base

class NumberConstantNode(base.MaxwellSimNode):
	node_type = ct.NodeType.NumberConstant
	bl_label = "Numerical Constant"
	
	input_socket_sets = {
		"Integer": {
			"Value": sockets.IntegerNumberSocketDef(),
		},
		"Rational": {
			"Value": sockets.RationalNumberSocketDef(),
		},
		"Real": {
			"Value": sockets.RealNumberSocketDef(),
		},
		"Complex": {
			"Value": sockets.ComplexNumberSocketDef(),
		},
	}
	output_socket_sets = input_socket_sets
	
	####################
	# - Callbacks
	####################
	@base.computes_output_socket(
		"Value",
		input_sockets={"Value"}
	)
	def compute_value(self, input_sockets) -> typ.Any:
		return input_sockets["Value"]



####################
# - Blender Registration
####################
BL_REGISTER = [
	NumberConstantNode,
]
BL_NODES = {
	ct.NodeType.NumberConstant: (
		ct.NodeCategory.MAXWELLSIM_INPUTS_CONSTANTS
	)
}
