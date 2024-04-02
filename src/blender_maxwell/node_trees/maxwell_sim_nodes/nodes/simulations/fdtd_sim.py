import sympy as sp
import tidy3d as td

from ... import contracts as ct
from ... import sockets
from .. import base


class FDTDSimNode(base.MaxwellSimNode):
	node_type = ct.NodeType.FDTDSim
	bl_label = 'FDTD Simulation'

	####################
	# - Sockets
	####################
	input_sockets = {
		'Domain': sockets.MaxwellSimDomainSocketDef(),
		'BCs': sockets.MaxwellBoundCondsSocketDef(),
		'Sources': sockets.MaxwellSourceSocketDef(
			is_list=True,
		),
		'Structures': sockets.MaxwellStructureSocketDef(
			is_list=True,
		),
		'Monitors': sockets.MaxwellMonitorSocketDef(
			is_list=True,
		),
	}
	output_sockets = {
		'FDTD Sim': sockets.MaxwellFDTDSimSocketDef(),
	}

	####################
	# - Output Socket Computation
	####################
	@base.computes_output_socket(
		'FDTD Sim',
		kind=ct.DataFlowKind.Value,
		input_sockets={'Sources', 'Structures', 'Domain', 'BCs', 'Monitors'},
	)
	def compute_fdtd_sim(self, input_sockets: dict) -> sp.Expr:
		sim_domain = input_sockets['Domain']
		sources = input_sockets['Sources']
		structures = input_sockets['Structures']
		bounds = input_sockets['BCs']
		monitors = input_sockets['Monitors']

		# if not isinstance(sources, list):
		# sources = [sources]
		# if not isinstance(structures, list):
		# structures = [structures]
		# if not isinstance(monitors, list):
		# monitors = [monitors]

		return td.Simulation(
			**sim_domain,  ## run_time=, size=, grid=, medium=
			structures=structures,
			sources=sources,
			monitors=monitors,
			boundary_spec=bounds,
		)


####################
# - Blender Registration
####################
BL_REGISTER = [
	FDTDSimNode,
]
BL_NODES = {ct.NodeType.FDTDSim: (ct.NodeCategory.MAXWELLSIM_SIMS)}
