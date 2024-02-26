import tidy3d as td
import sympy as sp
import sympy.physics.units as spu

from ... import contracts
from ... import sockets
from .. import base

class BoundBoxNode(base.MaxwellSimTreeNode):
	node_type = contracts.NodeType.BoundBox
	bl_label = "Bound Box"
	#bl_icon = ...
	
	####################
	# - Sockets
	####################
	input_sockets = {
		"x_pos": sockets.MaxwellBoundFaceSocketDef(
			label="+x",
		),
		"x_neg": sockets.MaxwellBoundFaceSocketDef(
			label="-x",
		),
		"y_pos": sockets.MaxwellBoundFaceSocketDef(
			label="+y",
		),
		"y_neg": sockets.MaxwellBoundFaceSocketDef(
			label="-y",
		),
		"z_pos": sockets.MaxwellBoundFaceSocketDef(
			label="+z",
		),
		"z_neg": sockets.MaxwellBoundFaceSocketDef(
			label="-z",
		),
	}
	output_sockets = {
		"bound": sockets.MaxwellBoundBoxSocketDef(
			label="Bound",
		),
	}
	
	####################
	# - Output Socket Computation
	####################
	@base.computes_output_socket("bound")
	def compute_simulation(self: contracts.NodeTypeProtocol) -> td.BoundarySpec:
		x_pos = self.compute_input("x_pos")
		x_neg = self.compute_input("x_neg")
		y_pos = self.compute_input("x_pos")
		y_neg = self.compute_input("x_neg")
		z_pos = self.compute_input("x_pos")
		z_neg = self.compute_input("x_neg")
		
		return td.BoundarySpec(
			x=td.Boundary(
				plus=x_pos,
				minus=x_neg,
			),
			y=td.Boundary(
				plus=y_pos,
				minus=y_neg,
			),
			z=td.Boundary(
				plus=z_pos,
				minus=z_neg,
			),
		)



####################
# - Blender Registration
####################
BL_REGISTER = [
	BoundBoxNode,
]
BL_NODES = {
	contracts.NodeType.BoundBox: (
		contracts.NodeCategory.MAXWELLSIM_BOUNDS
	)
}
