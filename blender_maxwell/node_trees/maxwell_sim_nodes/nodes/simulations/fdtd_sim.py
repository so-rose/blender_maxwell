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
		"run_time": sockets.PhysicalTimeSocketDef(
			label="Run Time",
		),
		"domain": sockets.PhysicalSize3DSocketDef(
			label="Domain",
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
			label="FDTD Sim",
		),
	}
	
	####################
	# - Output Socket Computation
	####################
	@base.computes_output_socket("fdtd_sim")
	def compute_simulation(self: contracts.NodeTypeProtocol) -> td.Simulation:
		_run_time = self.compute_input("run_time")
		_domain = self.compute_input("domain")
		ambient_medium = self.compute_input("ambient_medium")
		structures = [self.compute_input("structure")]
		sources = [self.compute_input("source")]
		bound = self.compute_input("bound")
		
		run_time = spu.convert_to(_run_time, spu.second) / spu.second
		domain = tuple(spu.convert_to(_domain, spu.um) / spu.um)
		
		return td.Simulation(
			size=domain,
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
