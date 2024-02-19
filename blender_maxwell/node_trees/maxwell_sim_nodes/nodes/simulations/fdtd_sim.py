import tidy3d as td
import sympy as sp
import sympy.physics.units as spu

from ... import contracts
from ... import sockets
from .. import base

class FDTDSimNode(base.MaxwellSimTreeNode):
	node_type = contracts.NodeType.FDTDSim
	
	bl_label = "FDTD Simulation"
	#bl_icon = ...
	
	####################
	# - Sockets
	####################
	input_sockets = {
		"run_time": sockets.RealNumberSocketDef(
			label="Run Time",
			default_value=1.0,
		),
		"size_x": sockets.RealNumberSocketDef(
			label="Size X",
			default_value=1.0,
		),
		"size_y": sockets.RealNumberSocketDef(
			label="Size Y",
			default_value=1.0,
		),
		"size_z": sockets.RealNumberSocketDef(
			label="Size Z",
			default_value=1.0,
		),
		"ambient_medium": sockets.MaxwellMediumSocketDef(
			label="Ambient Medium",
		),
		"source": sockets.MaxwellSourceSocketDef(
			label="Source",
		),
		"structure": sockets.MaxwellStructureSocketDef(
			label="Structure",
		),
		"bound": sockets.MaxwellBoundBoxSocketDef(
			label="Bound",
		),
	}
	output_sockets = {
		"fdtd_sim": sockets.MaxwellFDTDSimSocketDef(
			label="Medium",
		),
	}
	
	####################
	# - Output Socket Computation
	####################
	@base.computes_output_socket("fdtd_sim")
	def compute_simulation(self: contracts.NodeTypeProtocol) -> td.Simulation:
		run_time = self.compute_input("run_time")
		ambient_medium = self.compute_input("ambient_medium")
		structures = [self.compute_input("structure")]
		sources = [self.compute_input("source")]
		bound = self.compute_input("bound")
		
		size = (
			self.compute_input("size_x"),
			self.compute_input("size_y"),
			self.compute_input("size_z"),
		)
		
		return td.Simulation(
			size=size,
			medium=ambient_medium,
			structures=structures,
			sources=sources,
			boundary_spec=bound,
			run_time=run_time,
		)



####################
# - Blender Registration
####################
BL_REGISTER = [
	FDTDSimNode,
]
BL_NODES = {
	contracts.NodeType.FDTDSim: (
		contracts.NodeCategory.MAXWELLSIM_SIMS
	)
}
