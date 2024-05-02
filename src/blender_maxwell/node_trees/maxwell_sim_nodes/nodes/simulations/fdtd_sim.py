import typing as typ

import sympy as sp
import tidy3d as td

from ... import contracts as ct
from ... import sockets
from .. import base, events


class FDTDSimNode(base.MaxwellSimNode):
	node_type = ct.NodeType.FDTDSim
	bl_label = 'FDTD Simulation'

	####################
	# - Sockets
	####################
	input_sockets: typ.ClassVar = {
		'BCs': sockets.MaxwellBoundCondsSocketDef(),
		'Domain': sockets.MaxwellSimDomainSocketDef(),
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
	output_sockets: typ.ClassVar = {
		'Sim': sockets.MaxwellFDTDSimSocketDef(),
	}

	####################
	# - Output Socket Computation
	####################
	@events.computes_output_socket(
		'Sim',
		kind=ct.FlowKind.Value,
		input_sockets={'Sources', 'Structures', 'Domain', 'BCs', 'Monitors'},
		input_socket_kinds={
			'Sources': ct.FlowKind.Array,
			'Structures': ct.FlowKind.Array,
			'Domain': ct.FlowKind.Value,
			'BCs': ct.FlowKind.Value,
			'Monitors': ct.FlowKind.Array,
		},
	)
	def compute_fdtd_sim(self, input_sockets: dict) -> sp.Expr:
		## TODO: Visualize the boundary conditions on top of the sim domain
		sim_domain = input_sockets['Domain']
		sources = input_sockets['Sources']
		structures = input_sockets['Structures']
		bounds = input_sockets['BCs']
		monitors = input_sockets['Monitors']

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
