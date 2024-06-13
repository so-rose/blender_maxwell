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

import itertools
import typing as typ

import bpy
import jax
import numpy as np
import sympy as sp
import sympy.physics.units as spu
import tidy3d as td

from blender_maxwell.utils import bl_cache, logger, sim_symbols
from blender_maxwell.utils import sympy_extra as spux
from blender_maxwell.utils.frozendict import frozendict

from ... import contracts as ct
from ... import sockets
from .. import base, events

log = logger.get(__name__)

FK = ct.FlowKind
FS = ct.FlowSignal
MT = spux.MathType
PT = spux.PhysicalType

SimArray: typ.TypeAlias = frozendict[
	tuple[sim_symbols.SimSymbol, ...], tuple[typ.Any, ...], td.Simulation
]
SimArrayInfo: typ.TypeAlias = frozendict[
	tuple[sim_symbols.SimSymbol, ...], tuple[typ.Any, ...], td.Simulation
]


class RecomputeSimInfo(bpy.types.Operator):
	"""Recompute information about the simulation."""

	bl_idname = ct.OperatorType.NodeRecomputeSimInfo
	bl_label = 'Recompute Simuation Info'
	bl_description = (
		'Recompute information of a simulation attached to a `FDTDSimNode`.'
	)

	@classmethod
	def poll(cls, context):
		"""Allow running whenever a particular FDTDSim node is available."""
		return (
			# Check Tidy3DWebExporter is Accessible
			hasattr(context, 'node')
			and hasattr(context.node, 'node_type')
			and context.node.node_type == ct.NodeType.FDTDSim
		)

	def execute(self, context):
		"""Invalidate the `.sims` property, triggering reevaluation of all downstream information about the simulation."""
		node = context.node
		node.sims = bl_cache.Signal.InvalidateCache
		return {'FINISHED'}


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
			active_kind=FK.Array,
		),
		'Structures': sockets.MaxwellStructureSocketDef(
			active_kind=FK.Array,
		),
		'Monitors': sockets.MaxwellMonitorSocketDef(
			active_kind=FK.Array,
		),
	}
	output_socket_sets: typ.ClassVar = {
		'Single': {
			'Sim': sockets.MaxwellFDTDSimSocketDef(active_kind=FK.Value),
		},
		'Batch': {
			'Sims': sockets.MaxwellFDTDSimSocketDef(active_kind=FK.Array),
		},
		'Lazy': {
			'Sim': sockets.MaxwellFDTDSimSocketDef(active_kind=FK.Func),
		},
	}

	####################
	# - Properties: UI
	####################
	ui_limits: bool = bl_cache.BLField(False)
	ui_discretization: bool = bl_cache.BLField(False)
	ui_portability: bool = bl_cache.BLField(False)
	ui_propagation: bool = bl_cache.BLField(False)

	####################
	# - Properties: Simulation
	####################
	@bl_cache.cached_bl_property()
	def sims(self) -> SimArray | None:
		"""The complete description of all simulation output objects."""
		if self.active_socket_set == 'Single':
			sim_value = self.compute_output('Sim', kind=FK.Value)
			has_sim_value = not FS.check(sim_value)
			if has_sim_value:
				return {(): sim_value}

		elif self.active_socket_set == 'Batch':
			sim_array = self.compute_output('Sims', kind=FK.Array)
			has_sim_array = not FS.check(sim_array)
			if has_sim_array:
				return sim_array

		return None

	####################
	# - Properties: Propagation
	####################
	@bl_cache.cached_bl_property(depends_on={'sims'})
	def has_gain(self) -> SimArrayInfo | None:
		"""Whether any mediums in the simulation allow gain."""
		if self.sims is not None:
			return {k: sim.allow_gain for k, sim in self.sims.items()}
		return None

	@bl_cache.cached_bl_property(depends_on={'sims'})
	def has_complex_fields(self) -> SimArrayInfo | None:
		"""Whether complex fields are currently used in the simulation."""
		if self.sims is not None:
			return {k: sim.complex_fields for k, sim in self.sims.items()}
		return None

	@bl_cache.cached_bl_property(depends_on={'sims'})
	def min_wl(self) -> SimArrayInfo | None:
		"""The smallest wavelength that occurs in the simulation."""
		if self.sims is not None:
			return {k: sim.wvl_mat_min * spu.um for k, sim in self.sims.items()}
		return None

	####################
	# - Properties: Discretization
	####################
	@bl_cache.cached_bl_property(depends_on={'sims'})
	def time_range(self) -> SimArrayInfo | None:
		"""The time range of the simulation."""
		if self.sims is not None:
			return {
				k: sp.Matrix(
					[0, spu.convert_to(sim.run_time * spu.second, spu.picosecond)]
				)
				for k, sim in self.sims.items()
			}
		return None

	@bl_cache.cached_bl_property(depends_on={'sims'})
	def freq_range(self) -> SimArrayInfo | None:
		"""The total frequency range of the simulation, across all sources."""
		if self.sims is not None:
			return {
				k: spu.convert_to(
					sp.Matrix([sim.frequency_range[0], sim.frequency_range[1]])
					* spu.hertz,
					spux.THz,
				)
				for k, sim in self.sims.items()
			}
		return None

	@bl_cache.cached_bl_property(depends_on={'sims'})
	def time_step(self) -> SimArrayInfo | None:
		"""The time step of the simulation."""
		if self.sims is not None:
			return {k: sim.dt * spu.second for k, sim in self.sims.items()}
		return None

	@bl_cache.cached_bl_property(depends_on={'sims'})
	def time_steps(self) -> SimArrayInfo | None:
		"""The time step of the simulation."""
		if self.sims is not None:
			return {k: sim.num_time_steps for k, sim in self.sims.items()}
		return None

	@bl_cache.cached_bl_property(depends_on={'sims'})
	def nyquist_step(self) -> SimArrayInfo | None:
		"""The number of time-steps needed to theoretically provide for correctly resolved sampling of the simulation grid."""
		if self.sims is not None:
			return {k: sim.nyquist_step for k, sim in self.sims.items()}
		return None

	@bl_cache.cached_bl_property(depends_on={'sims'})
	def num_cells(self) -> SimArrayInfo | None:
		"""The number of 3D cells for which the simulation is discretized."""
		if self.sims is not None:
			return {k: sim.num_cells for k, sim in self.sims.items()}
		return None

	####################
	# - Properties: Data
	####################
	@bl_cache.cached_bl_property(depends_on={'sims'})
	def monitor_data_sizes(self) -> SimArrayInfo | None:
		"""The total data expected to be taken by each monitors."""
		if self.sims is not None:
			return {k: sim.monitors_data_size for k, sim in self.sims.items()}
		return None

	@bl_cache.cached_bl_property(depends_on={'monitor_data_sizes'})
	def total_monitor_data_size(self) -> SimArrayInfo | None:
		"""The total data taken by the monitors."""
		if self.monitor_data_sizes is not None:
			return {
				k: sum(sizes.values()) for k, sizes in self.monitor_data_sizes.items()
			}
		return None

	####################
	# - Properties: Lists
	####################
	@bl_cache.cached_bl_property(depends_on={'sims'})
	def list_datasets(self) -> SimArrayInfo | None:
		"""List of custom datasets required by the simulation."""
		if self.sims is not None:
			return {k: sim.custom_datasets for k, sim in self.sims.items()}
		return None

	@bl_cache.cached_bl_property(depends_on={'sims'})
	def list_vol_structures(self) -> SimArrayInfo | None:
		"""List of volumetric structures, where 2D mediums were converted to 3D."""
		if self.sims is not None:
			return {k: sim.volumetric_structures for k, sim in self.sims.items()}
		return None

	####################
	# - Properties: Validated
	####################
	@bl_cache.cached_bl_property(depends_on={'sims'})
	def sims_valid(self) -> SimArrayInfo | None:
		"""Whether all sims are valid."""
		if self.sims is not None:
			validity = {}
			for k, sim in self.sims.items():  # noqa: B007
				try:
					pass  ## TODO: VERY slow, batch checking is infeasible
					# sim.validate_pre_upload(source_required=True)
				except td.exceptions.SetupError:
					validity[k] = False
				else:
					validity[k] = True

			return validity
		return None

	####################
	# - Info
	####################
	@bl_cache.cached_bl_property(
		depends_on={
			'sims',
			'time_range',
			'freq_range',
			'min_wl',
			'num_cells',
			'time_steps',
			'nyquist_steps',
			'time_step',
			'sims_valid',
			'total_monitor_data_size',
			'has_gain',
			'has_complex_fields',
			'ui_limits',
			'ui_discretization',
			'ui_portability',
			'ui_propagation',
		}
	)
	def sim_labels(self) -> SimArrayInfo | None:
		"""Pre-processed labels for efficient drawing of simulation info."""
		if self.sims is not None:
			sims_vals_labels = {}
			for syms_vals in self.sims:
				labels = []

				if syms_vals:
					labels += [
						'|'.join(
							[
								f'{sym.name_pretty}={val:,.2f}'
								for sym, val in zip(*syms_vals, strict=True)
							]
						),
					]

				labels += [
					['Limits', 'ui_limits'],
				]
				if self.ui_limits:
					labels += [
						('max t', spux.sp_to_str(self.time_range[syms_vals][1])),
						('min f', spux.sp_to_str(self.freq_range[syms_vals][0])),
						('max f', spux.sp_to_str(self.freq_range[syms_vals][1])),
						('min λ', spux.sp_to_str(self.min_wl[syms_vals].n(2))),
					]

				labels += [
					['Discretization', 'ui_discretization'],
				]
				if self.ui_discretization:
					labels += [
						('cells', f'{self.num_cells[syms_vals]:,}'),
						('num Δt', f'{self.time_steps[syms_vals]:,}'),
						('nyq Δt', f'{self.nyquist_step[syms_vals]}'),
						('Δt', spux.sp_to_str(self.time_step[syms_vals].n(2))),
					]

				labels += [
					['Portability', 'ui_portability'],
				]
				if self.ui_portability:
					labels += [
						('Valid?', str(self.sims_valid[syms_vals])),
						(
							'Σ mon',
							f'{self.total_monitor_data_size[syms_vals] / 1000000:,.2f}MB',
						),
					]

				labels += [
					['Propagation', 'ui_propagation'],
				]
				if self.ui_propagation:
					labels += [
						('Gain?', str(self.has_gain[syms_vals])),
						('𝐄𝐇 ∈ ℂ', str(self.has_complex_fields[syms_vals])),
					]

				sims_vals_labels[syms_vals] = labels
			return sims_vals_labels
		return None

	def draw_info(self, _, layout: bpy.types.UILayout):
		"""Draw information about the simulation, if any."""
		row = layout.row(align=True)
		row.alignment = 'CENTER'
		row.label(text='Sim Info')
		row.operator(ct.OperatorType.NodeRecomputeSimInfo, icon='FILE_REFRESH', text='')

		# Simulation Info
		if self.sim_labels is not None:
			for labels in self.sim_labels.values():
				box = layout.box()
				for el in labels:
					# Header
					if isinstance(el, list):
						row = box.row(align=True)
						# row.alignment = 'EXPAND'
						row.prop(self, self.blfields[el[1]], text=el[0], toggle=True)
						# row.label(text=el)

						split = box.split(factor=0.4)
						col_l = split.column(align=True)
						col_r = split.column(align=True)

					# Label Pair
					elif isinstance(el, tuple):
						col_l.label(text=el[0])
						col_r.label(text=el[1])

					else:
						raise TypeError

				break

	####################
	# - Events
	####################
	@events.on_value_changed(
		# Trigger
		socket_name={'BCs', 'Domain', 'Sources', 'Structures', 'Monitors'},
		prop_name={'active_socket_set'},
		# Loaded
		props={'active_socket_set'},
		output_sockets={'Sim'},
		outscks_kinds={'Sim': FK.Params},
	)
	def on_any_changed(self, props, output_sockets) -> None:
		"""Manage loose input sockets in response to symbolic simulation elements."""
		# Loose Input Sockets
		output_params = output_sockets['Sim']
		active_socket_set = props['active_socket_set']
		if (
			active_socket_set in ['Single', 'Batch']
			and output_params.symbols
			and set(self.loose_input_sockets)
			!= {sym.name for sym in output_params.symbols}
		):
			self.loose_input_sockets = {
				sym.name: sockets.ExprSocketDef(
					**(
						expr_info
						| {
							'active_kind': FK.Value,
							'use_value_range_swapper': (active_socket_set == 'Batch'),
						}
					)
				)
				for sym, expr_info in output_params.sym_expr_infos.items()
			}

		elif self.loose_input_sockets:
			self.loose_input_sockets = {}

	####################
	# - FlowKind.Value
	####################
	@events.computes_output_socket(
		'Sim',
		kind=FK.Value,
		# Loaded
		props={'active_socket_set'},
		outscks_kinds={'Sim': {FK.Func, FK.Params}},
		all_loose_input_sockets=True,
		loose_input_sockets_kind={FK.Func, FK.Params},
	)
	def compute_value(
		self, props, loose_input_sockets, output_sockets
	) -> td.Simulation | FS:
		"""Compute the particular value of the simulation domain from strictly non-symbolic inputs."""
		func = output_sockets['Sim'][FK.Func]
		params = output_sockets['Sim'][FK.Params]

		has_func = not FS.check(func)
		has_params = not FS.check(params)

		active_socket_set = props['active_socket_set']
		if has_func and has_params and active_socket_set == 'Single':
			symbol_values = {
				sym: events.realize_known(loose_input_sockets[sym.name])
				for sym in params.sorted_symbols
			}

			return func.realize(
				params,
				symbol_values=frozendict(
					{
						sym: events.realize_known(loose_input_sockets[sym.name])
						for sym in params.sorted_symbols
					}
				),
			).updated_copy(
				attrs=dict(
					ct.SimMetadata(
						realizations=ct.SimRealizations(
							syms=tuple(symbol_values.keys()),
							vals=tuple(symbol_values.values()),
						)
					)
				)
			)
		return FS.FlowPending

	####################
	# - FlowKind.Array
	####################
	@events.computes_output_socket(
		'Sims',
		kind=FK.Array,
		# Loaded
		props={'active_socket_set'},
		outscks_kinds={'Sim': {FK.Func, FK.Params}},
		all_loose_input_sockets=True,
		loose_input_sockets_kind={FK.Func, FK.Params},
	)
	def compute_array(
		self, props, loose_input_sockets, output_sockets
	) -> SimArray | FS:
		"""Produce a batch of simulations as a dictionary, indexed by a twice-nested tuple matching a `SimSymbol` tuple to a corresponding tuple of values."""
		func = output_sockets['Sim'][FK.Func]
		params = output_sockets['Sim'][FK.Params]

		active_socket_set = props['active_socket_set']
		if active_socket_set == 'Batch':
			# Realize Values per-Symbol
			## -> First, we realize however many values requested per symbol.
			sym_datas: dict[sim_symbols.SimSymbol, list] = {}
			for sym in params.sorted_symbols:
				if sym.name not in loose_input_sockets:
					return FS.FlowPending

				# Realize Data for Symbol
				## -> This may be a single scalar/vector/matrix.
				## -> This may also be many _scalars_.
				sym_data = events.realize_known(
					loose_input_sockets[sym.name],
					freeze=True,
				)
				if sym_data is None:
					return FS.FlowPending

				# Single Value per-Symbol
				if sym.shape_len == 0:
					sym_datas |= {sym: (sym_data,)}

				# Many Values per-Symbol
				else:
					sym_datas |= {sym: sym_data}

			# Realize Function per-Combination
			## -> td.Simulation requires single, specific values for all syms.
			## -> What we have is many specific values for each sym.
			## -> With a single comprehension, we resolve this difference.
			## -> The end-result is an annotated td.Simulation per-combo.
			## -> NOTE: This might be big! Such are parameter-sweeps.
			syms = tuple(sym_datas.keys())
			return {
				(syms, vals): func.realize(
					params,
					symbol_values=frozendict(zip(syms, vals, strict=True)),
				).updated_copy(
					attrs=dict(
						ct.SimMetadata(
							realizations=ct.SimRealizations(syms=syms, vals=vals)
						)
					)
				)
				for vals in itertools.product(*sym_datas.values())
			}

		return FS.FlowPending

	####################
	# - FlowKind.Func
	####################
	@events.computes_output_socket(
		'Sim',
		kind=FK.Func,
		# Loaded
		inscks_kinds={
			'BCs': FK.Func,
			'Domain': FK.Func,
			'Sources': FK.Func,
			'Structures': FK.Func,
			'Monitors': FK.Func,
		},
	)
	def compute_fdtd_sim_func(self, input_sockets) -> ct.FuncFlow:
		"""Compute a single simulation, given that all inputs are non-symbolic."""
		bounds = input_sockets['BCs']
		sim_domain = input_sockets['Domain']
		sources = input_sockets['Sources']
		structures = input_sockets['Structures']
		monitors = input_sockets['Monitors']

		return (bounds | sim_domain | sources | structures | monitors).compose_within(
			lambda els: td.Simulation(
				boundary_spec=els[0],
				**els[1],
				sources=els[2],
				structures=els[3],
				monitors=els[4],
			),
		)

	####################
	# - FlowKind.Params
	####################
	@events.computes_output_socket(
		'Sim',
		kind=FK.Params,
		# Loaded
		inscks_kinds={
			'BCs': FK.Params,
			'Domain': FK.Params,
			'Sources': FK.Params,
			'Structures': FK.Params,
			'Monitors': FK.Params,
		},
	)
	def compute_params(self, input_sockets) -> td.Simulation | FS:
		"""Compute all function parameters needed to create the simulation."""
		# Compute Output Parameters
		bounds = input_sockets['BCs']
		sim_domain = input_sockets['Domain']
		sources = input_sockets['Sources']
		structures = input_sockets['Structures']
		monitors = input_sockets['Monitors']

		return bounds | sim_domain | sources | structures | monitors


####################
# - Blender Registration
####################
BL_REGISTER = [
	RecomputeSimInfo,
	FDTDSimNode,
]
BL_NODES = {ct.NodeType.FDTDSim: (ct.NodeCategory.MAXWELLSIM_SIMS)}
