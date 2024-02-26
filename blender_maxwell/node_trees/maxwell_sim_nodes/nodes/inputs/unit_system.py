import bpy
import sympy as sp

from ... import contracts
from ... import sockets
from .. import base

class PhysicalUnitSystemNode(base.MaxwellSimTreeNode):
	node_type = contracts.NodeType.UnitSystem
	
	bl_label = "Unit System Constant"
	
	input_sockets = {
		"unit_system": sockets.PhysicalUnitSystemSocketDef(
			label="Unit System",
			show_by_default=True,
		),
	}
	output_sockets = {
		"unit_system": sockets.PhysicalUnitSystemSocketDef(
			label="Unit System",
		),
	}
	
	####################
	# - Callbacks
	####################
	@base.computes_output_socket("unit_system")
	def compute_value(self: contracts.NodeTypeProtocol) -> sp.Expr:
		return self.compute_input("unit_system")



####################
# - Blender Registration
####################
BL_REGISTER = [
	PhysicalUnitSystemNode,
]
BL_NODES = {
	contracts.NodeType.UnitSystem: (
		contracts.NodeCategory.MAXWELLSIM_INPUTS
	)
}
