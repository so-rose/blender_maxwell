# blender_maxwell
# Copyright (C) 2024 blender_maxwell Project Contributors
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

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
		if any(ct.FlowSignal.check(inp) for inp in input_sockets):
			return ct.FlowSignal.FlowPending

		sim_domain = input_sockets['Domain']
		sources = input_sockets['Sources']
		structures = input_sockets['Structures']
		bounds = input_sockets['BCs']
		monitors = input_sockets['Monitors']
		return td.Simulation(
			**sim_domain,
			structures=structures,
			sources=sources,
			monitors=monitors,
			boundary_spec=bounds,
		)
		## TODO: Visualize the boundary conditions on top of the sim domain


####################
# - Blender Registration
####################
BL_REGISTER = [
	FDTDSimNode,
]
BL_NODES = {ct.NodeType.FDTDSim: (ct.NodeCategory.MAXWELLSIM_SIMS)}
