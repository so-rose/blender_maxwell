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

"""Implements `FDTDSimNode`."""

import typing as typ

import bpy
import tidy3d as td
import tidy3d.plugins.adjoint as tdadj

from blender_maxwell.utils import bl_cache, logger

from ... import contracts as ct
from ... import sockets
from .. import base, events

log = logger.get(__name__)


class FDTDSimNode(base.MaxwellSimNode):
	"""Definition of a complete FDTD simulation, including boundary conditions, domain, sources, structures, monitors, and other configuration."""

	node_type = ct.NodeType.FDTDSim
	bl_label = 'FDTD Simulation'

	####################
	# - Sockets
	####################
	input_sockets: typ.ClassVar = {
		'BCs': sockets.MaxwellBoundCondsSocketDef(),
		'Domain': sockets.MaxwellSimDomainSocketDef(),
		'Sources': sockets.MaxwellSourceSocketDef(
			active_kind=ct.FlowKind.Array,
		),
		'Structures': sockets.MaxwellStructureSocketDef(
			active_kind=ct.FlowKind.Array,
		),
		'Monitors': sockets.MaxwellMonitorSocketDef(
			active_kind=ct.FlowKind.Array,
		),
	}
	output_socket_sets: typ.ClassVar = {
		'Single': {
			'Sim': sockets.MaxwellFDTDSimSocketDef(active_kind=ct.FlowKind.Value),
		},
		'Batch': {
			'Sim': sockets.MaxwellFDTDSimSocketDef(active_kind=ct.FlowKind.Array),
		},
		'Lazy': {
			'Sim': sockets.MaxwellFDTDSimSocketDef(active_kind=ct.FlowKind.Func),
		},
	}

	####################
	# - Properties
	####################
	differentiable: bool = bl_cache.BLField(False)

	####################
	# - UI
	####################
	def draw_props(self, _: bpy.types.Context, layout: bpy.types.UILayout):
		layout.prop(
			self,
			self.blfields['differentiable'],
			text='Differentiable',
			toggle=True,
		)

	####################
	# - Events
	####################
	@events.on_value_changed(
		# Trigger
		socket_name={'Sources', 'Structures', 'Domain', 'BCs', 'Monitors'},
		run_on_init=True,
		# Loaded
		props={'active_socket_set'},
		output_sockets={'Sim'},
		output_socket_kinds={'Sim': ct.FlowKind.Params},
	)
	def on_any_changed(self, props, output_sockets) -> None:
		"""Create loose input sockets."""
		params = output_sockets['Sim']
		has_params = not ct.FlowSignal.check(params)

		# Declare Loose Sockets that Realize Symbols
		## -> This happens if Params contains not-yet-realized symbols.
		active_socket_set = props['active_socket_set']
		if active_socket_set in ['Value', 'Batch'] and has_params and params.symbols:
			if set(self.loose_input_sockets) != {sym.name for sym in params.symbols}:
				self.loose_input_sockets = {
					sym.name: sockets.ExprSocketDef(
						**(
							expr_info
							| {
								'active_kind': ct.FlowKind.Value,
								'use_value_range_swapper': (
									active_socket_set == 'Value'
								),
							}
						)
					)
					for sym, expr_info in params.sym_expr_infos.items()
				}

		elif self.loose_input_sockets:
			self.loose_input_sockets = {}

	####################
	# - FlowKind.Value
	####################
	@events.computes_output_socket(
		'Sim',
		kind=ct.FlowKind.Value,
		# Loaded
		props={'differentiable'},
		input_sockets={'Sources', 'Structures', 'Domain', 'BCs', 'Monitors'},
		input_socket_kinds={
			'Sources': ct.FlowKind.Array,
			'Structures': ct.FlowKind.Array,
			'Monitors': ct.FlowKind.Array,
		},
		output_sockets={'Sim'},
		output_socket_kinds={'Sim': ct.FlowKind.Params},
	)
	def compute_fdtd_sim_value(
		self, props, input_sockets, output_sockets
	) -> td.Simulation | tdadj.JaxSimulation | ct.FlowSignal:
		"""Compute a single FDTD simulation definition, so long as the inputs are neither symbolic or differentiable."""
		sim_domain = input_sockets['Domain']
		sources = input_sockets['Sources']
		structures = input_sockets['Structures']
		bounds = input_sockets['BCs']
		monitors = input_sockets['Monitors']
		output_params = output_sockets['Sim']

		has_sim_domain = not ct.FlowSignal.check(sim_domain)
		has_sources = not ct.FlowSignal.check(sources)
		has_structures = not ct.FlowSignal.check(structures)
		has_bounds = not ct.FlowSignal.check(bounds)
		has_monitors = not ct.FlowSignal.check(monitors)
		has_output_params = not ct.FlowSignal.check(output_params)

		differentiable = props['differentiable']
		if (
			has_sim_domain
			and has_sources
			and has_structures
			and has_bounds
			and has_monitors
			and has_output_params
			and not differentiable
		):
			return td.Simulation(
				**sim_domain,
				sources=sources,
				structures=structures,
				boundary_spec=bounds,
				monitors=monitors,
			)
		return ct.FlowSignal.FlowPending

	####################
	# - FlowKind.Func
	####################
	@events.computes_output_socket(
		'Sim',
		kind=ct.FlowKind.Func,
		# Loaded
		props={'differentiable'},
		input_sockets={'Sources', 'Structures', 'Domain', 'BCs', 'Monitors'},
		input_socket_kinds={
			'Sources': ct.FlowKind.Func,
			'Structures': ct.FlowKind.Func,
			'Monitors': ct.FlowKind.Func,
		},
		output_sockets={'Sim'},
		output_socket_kinds={'Sim': ct.FlowKind.Params},
	)
	def compute_fdtd_sim_func(
		self, props, input_sockets, output_sockets
	) -> td.Simulation | tdadj.JaxSimulation | ct.FlowSignal:
		"""Compute a single simulation, given that all inputs are non-symbolic."""
		sim_domain = input_sockets['Domain']
		sources = input_sockets['Sources']
		structures = input_sockets['Structures']
		bounds = input_sockets['BCs']
		monitors = input_sockets['Monitors']
		output_params = output_sockets['Sim']

		has_sim_domain = not ct.FlowSignal.check(sim_domain)
		has_sources = not ct.FlowSignal.check(sources)
		has_structures = not ct.FlowSignal.check(structures)
		has_bounds = not ct.FlowSignal.check(bounds)
		has_monitors = not ct.FlowSignal.check(monitors)
		has_output_params = not ct.FlowSignal.check(output_params)

		if (
			has_sim_domain
			and has_sources
			and has_structures
			and has_bounds
			and has_monitors
			and has_output_params
		):
			differentiable = props['differentiable']
			if differentiable:
				return (
					sim_domain | sources | structures | bounds | monitors
				).compose_within(
					enclosing_func=lambda els: tdadj.JaxSimulation(
						**els[0],
						sources=els[1],
						structures=els[2]['static'],
						input_structures=els[2]['differentiable'],
						boundary_spec=els[3],
						monitors=els[4]['static'],
						output_monitors=els[4]['differentiable'],
					),
					supports_jax=True,
				)
			return (
				sim_domain | sources | structures | bounds | monitors
			).compose_within(
				enclosing_func=lambda els: td.Simulation(
					**els[0],
					sources=els[1],
					structures=els[2],
					boundary_spec=els[3],
					monitors=els[4],
				),
				supports_jax=False,
			)
		return ct.FlowSignal.FlowPending

	####################
	# - FlowKind.Params
	####################
	@events.computes_output_socket(
		'Sim',
		kind=ct.FlowKind.Params,
		# Loaded
		props={'differentiable'},
		input_sockets={'Sources', 'Structures', 'Domain', 'BCs', 'Monitors'},
		input_socket_kinds={
			'Sources': ct.FlowKind.Params,
			'Structures': ct.FlowKind.Params,
			'Monitors': ct.FlowKind.Params,
		},
	)
	def compute_fdtd_sim_params(
		self, props, input_sockets
	) -> td.Simulation | tdadj.JaxSimulation | ct.FlowSignal:
		"""Compute a single simulation, given that all inputs are non-symbolic."""
		sim_domain = input_sockets['Domain']
		sources = input_sockets['Sources']
		structures = input_sockets['Structures']
		bounds = input_sockets['BCs']
		monitors = input_sockets['Monitors']

		has_sim_domain = not ct.FlowSignal.check(sim_domain)
		has_sources = not ct.FlowSignal.check(sources)
		has_structures = not ct.FlowSignal.check(structures)
		has_bounds = not ct.FlowSignal.check(bounds)
		has_monitors = not ct.FlowSignal.check(monitors)

		if (
			has_sim_domain
			and has_sources
			and has_structures
			and has_bounds
			and has_monitors
		):
			# Determine Differentiable Match
			## -> 'structures' is diff when **any** are diff.
			## -> 'monitors' is also diff when **any** are diff.
			## -> Only parameters through diff structs can be diff'ed by.
			## -> Similarly, only diff monitors will have gradients computed.
			return sim_domain | sources | structures | bounds | monitors
		return ct.FlowSignal.FlowPending


####################
# - Blender Registration
####################
BL_REGISTER = [
	FDTDSimNode,
]
BL_NODES = {ct.NodeType.FDTDSim: (ct.NodeCategory.MAXWELLSIM_SIMS)}
