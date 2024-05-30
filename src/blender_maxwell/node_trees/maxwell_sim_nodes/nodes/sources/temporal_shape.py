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

"""Implements the `TemporalShapeNode`."""

import enum
import typing as typ

import bpy
import numpy as np
import sympy as sp
import sympy.physics.units as spu
import tidy3d as td
from tidy3d.components.data.data_array import TimeDataArray as td_TimeDataArray
from tidy3d.components.data.dataset import TimeDataset as td_TimeDataset

from blender_maxwell.utils import bl_cache, logger, sim_symbols
from blender_maxwell.utils import extra_sympy_units as spux

from ... import contracts as ct
from ... import managed_objs, sockets
from .. import base, events

log = logger.get(__name__)


# Select Default Time Unit for Envelope
## -> Chosen to align with the default envelope_time_unit.
## -> This causes it to be correct from the start.
t_def = sim_symbols.t(spux.PhysicalType.Time.valid_units[0])


class TemporalShapeNode(base.MaxwellSimNode):
	"""Declare a source-time dependence for use in simulation source nodes."""

	node_type = ct.NodeType.TemporalShape
	bl_label = 'Temporal Shape'

	####################
	# - Sockets
	####################
	input_sockets: typ.ClassVar = {
		'μ Freq': sockets.ExprSocketDef(
			physical_type=spux.PhysicalType.Freq,
			default_unit=spux.THz,
			default_value=500,
		),
		'σ Freq': sockets.ExprSocketDef(
			physical_type=spux.PhysicalType.Freq,
			default_unit=spux.THz,
			default_value=200,
		),
		'max E': sockets.ExprSocketDef(
			mathtype=spux.MathType.Complex,
			physical_type=spux.PhysicalType.EField,
			default_value=1 + 0j,
		),
		'Offset Time': sockets.ExprSocketDef(default_value=5, abs_min=2.5),
	}
	input_socket_sets: typ.ClassVar = {
		'Pulse': {
			'Remove DC': sockets.BoolSocketDef(default_value=True),
		},
		'Constant': {},
		'Symbolic': {
			't Range': sockets.ExprSocketDef(
				active_kind=ct.FlowKind.Range,
				physical_type=spux.PhysicalType.Time,
				default_unit=spu.picosecond,
				default_min=0,
				default_max=10,
				default_steps=100,
			),
			'Envelope': sockets.ExprSocketDef(
				default_symbols=[t_def],
				default_value=10 * t_def.sp_symbol,
			),
		},
	}

	output_sockets: typ.ClassVar = {
		'Temporal Shape': sockets.MaxwellTemporalShapeSocketDef(),
	}

	managed_obj_types: typ.ClassVar = {
		'plot': managed_objs.ManagedBLImage,
	}

	####################
	# - Properties
	####################
	active_envelope_time_unit: enum.StrEnum = bl_cache.BLField(
		enum_cb=lambda self, _: self.search_time_units(),
	)

	def search_time_units(self) -> list[ct.BLEnumElement]:
		"""Compute all valid time units."""
		return [
			(sp.sstr(unit), spux.sp_to_str(unit), sp.sstr(unit), '', i)
			for i, unit in enumerate(spux.PhysicalType.Time.valid_units)
		]

	@bl_cache.cached_bl_property(depends_on={'active_envelope_time_unit'})
	def envelope_time_unit(self) -> spux.Unit | None:
		"""Gets the current active unit for the envelope time symbol.

		Returns:
			The current active `sympy` unit.

			If the socket expression is unitless, this returns `None`.
		"""
		if self.active_envelope_time_unit is not None:
			return spux.unit_str_to_unit(self.active_envelope_time_unit)

		return None

	####################
	# - UI
	####################
	def draw_props(self, _: bpy.types.Context, layout: bpy.types.UILayout):
		if (
			self.active_socket_set == 'Symbolic'
			and self.inputs.get('Envelope')
			and not self.inputs['Envelope'].is_linked
		):
			row = layout.row()
			row.alignment = 'CENTER'
			row.label(text='Envelope Time Unit')

			row = layout.row()
			row.prop(
				self,
				self.blfields['active_envelope_time_unit'],
				text='',
				toggle=True,
			)

	def draw_info(self, _: bpy.types.Context, layout: bpy.types.UILayout) -> None:
		if self.active_socket_set != 'Symbolic':
			box = layout.box()
			row = box.row()
			row.alignment = 'CENTER'
			row.label(text='Parameter Scale')

			# Split
			split = box.split(factor=0.3, align=False)

			## LHS: Parameter Names
			col = split.column()
			col.alignment = 'RIGHT'
			col.label(text='Off t:')

			## RHS: Parameter Units
			col = split.column()
			col.label(text='1 / 2π·σ(𝑓)')

	####################
	# - Events
	####################
	@events.on_value_changed(
		# Trigger
		prop_name={'active_socket_set', 'envelope_time_unit'},
		# Loaded
		props={'active_socket_set', 'envelope_time_unit'},
	)
	def on_envelope_time_unit_changed(self, props) -> None:
		"""Ensure the envelope expression's time symbol has the time unit defined by the node."""
		active_socket_set = props['active_socket_set']
		envelope_time_unit = props['envelope_time_unit']
		if active_socket_set == 'Symbolic':
			bl_socket = self.inputs['Envelope']
			wanted_t_sym = sim_symbols.t(envelope_time_unit)

			if not bl_socket.symbols or bl_socket.symbols[0] != wanted_t_sym:
				bl_socket.symbols = [wanted_t_sym]

	####################
	# - FlowKind.Value
	####################
	@events.computes_output_socket(
		'Temporal Shape',
		kind=ct.FlowKind.Value,
		# Loaded
		output_sockets={'Temporal Shape'},
		output_socket_kinds={'Temporal Shape': {ct.FlowKind.Func, ct.FlowKind.Params}},
	)
	def compute_domain_value(self, output_sockets) -> ct.ParamsFlow | ct.FlowSignal:
		"""Compute a single temporal shape."""
		output_func = output_sockets['Temporal Shape'][ct.FlowKind.Func]
		output_params = output_sockets['Temporal Shape'][ct.FlowKind.Params]

		has_output_func = not ct.FlowSignal.check(output_func)
		has_output_params = not ct.FlowSignal.check(output_params)

		if has_output_func and has_output_params and not output_params.symbols:
			return output_func.realize(output_params)
		return ct.FlowSignal.FlowPending

	####################
	# - FlowKind: Func
	####################
	@events.computes_output_socket(
		'Temporal Shape',
		kind=ct.FlowKind.Func,
		# Loaded
		props={'active_socket_set'},
		input_sockets={
			'max E',
			'μ Freq',
			'σ Freq',
			'Offset Time',
			'Remove DC',
			't Range',
			'Envelope',
		},
		input_socket_kinds={
			'max E': ct.FlowKind.Func,
			'μ Freq': ct.FlowKind.Func,
			'σ Freq': ct.FlowKind.Func,
			'Offset Time': ct.FlowKind.Func,
			'Remove DC': ct.FlowKind.Value,
			't Range': ct.FlowKind.Func,
			'Envelope': {ct.FlowKind.Func, ct.FlowKind.Params},
		},
	)
	def compute_temporal_shape_func(
		self,
		props,
		input_sockets,
	) -> td.GaussianPulse:
		"""Compute a single temporal shape from non-parameterized inputs."""
		mean_freq = input_sockets['μ Freq']
		std_freq = input_sockets['σ Freq']
		max_e = input_sockets['max E']
		offset = input_sockets['Offset Time']

		has_mean_freq = not ct.FlowSignal.check(mean_freq)
		has_std_freq = not ct.FlowSignal.check(std_freq)
		has_max_e = not ct.FlowSignal.check(max_e)
		has_offset = not ct.FlowSignal.check(offset)

		if has_mean_freq and has_std_freq and has_max_e and has_offset:
			common_func = (
				max_e.scale_to_unit_system(ct.UNITS_TIDY3D)
				| mean_freq.scale_to_unit_system(ct.UNITS_TIDY3D)
				| std_freq.scale_to_unit_system(ct.UNITS_TIDY3D)
				| offset  ## Already unitless
			)
			match props['active_socket_set']:
				case 'Pulse':
					remove_dc = input_sockets['Remove DC']

					has_remove_dc = not ct.FlowSignal.check(remove_dc)

					if has_remove_dc:
						return common_func.compose_within(
							lambda els: td.GaussianPulse(
								amplitude=complex(els[0]).real,
								phase=complex(els[0]).imag,
								freq0=els[1],
								fwidth=els[2],
								offset=els[3],
								remove_dc_component=remove_dc,
							),
						)

				case 'Constant':
					return common_func.compose_within(
						lambda els: td.GaussianPulse(
							amplitude=complex(els[0]).real,
							phase=complex(els[0]).imag,
							freq0=els[1],
							fwidth=els[2],
							offset=els[3],
						),
					)

				case 'Symbolic':
					t_range = input_sockets['t Range']
					envelope = input_sockets['Envelope'][ct.FlowKind.Func]
					envelope_params = input_sockets['Envelope'][ct.FlowKind.Params]

					has_t_range = not ct.FlowSignal.check(t_range)
					has_envelope = not ct.FlowSignal.check(envelope)
					has_envelope_params = not ct.FlowSignal.check(envelope_params)

					if (
						has_t_range
						and has_envelope
						and has_envelope_params
						and len(envelope_params.symbols) == 1
						## TODO: Allow unrealized envelope symbols
						and any(
							sym.physical_type is spux.PhysicalType.Time
							for sym in envelope_params.symbols
						)
					):
						envelope_time_unit = next(
							sym.unit
							for sym in envelope_params.symbols
							if sym.physical_type is spux.PhysicalType.Time
						)

						# Deduce Partially Realized Envelope Function
						## -> We need a pure-numerical function w/pre-realized stuff baked in.
						## -> 'realize_partial' does this for us.
						envelope_realizer = envelope.realize_partial(envelope_params)

						# Compose w/Envelope Function
						## -> First, the numerical time values must be converted.
						## -> This ensures that the raw array is compatible w/the envelope.
						## -> Then, we can compose w/the purely numerical 'envelope_realizer'.
						## -> Because of the checks, we've guaranteed that all this is correct.
						return (
							common_func  ## 1 | freq0, 2 | fwidth, 3 | offset
							| t_range.scale_to_unit_system(ct.UNITS_TIDY3D)  ## 4
							| t_range.scale_to_unit(envelope_time_unit).compose_within(
								lambda t: envelope_realizer(t)
							)  ## 5
						).compose_within(
							lambda els: td.CustomSourceTime(
								amplitude=complex(els[0]).real,
								phase=complex(els[0]).imag,
								freq0=els[1],
								fwidth=els[2],
								offset=els[3],
								source_time_dataset=td_TimeDataset(
									values=td_TimeDataArray(
										els[5], coords={'t': np.array(els[4])}
									)
								),
							)
						)

		return ct.FlowSignal.FlowPending

	####################
	# - FlowKind: Params
	####################
	@events.computes_output_socket(
		'Temporal Shape',
		kind=ct.FlowKind.Params,
		# Loaded
		props={'active_socket_set', 'envelope_time_unit'},
		input_sockets={
			'max E',
			'μ Freq',
			'σ Freq',
			'Offset Time',
			't Range',
		},
		input_socket_kinds={
			'max E': ct.FlowKind.Params,
			'μ Freq': ct.FlowKind.Params,
			'σ Freq': ct.FlowKind.Params,
			'Offset Time': ct.FlowKind.Params,
			't Range': ct.FlowKind.Params,
		},
	)
	def compute_temporal_shape_params(
		self,
		props,
		input_sockets,
	) -> td.GaussianPulse:
		"""Compute a single temporal shape from non-parameterized inputs."""
		mean_freq = input_sockets['μ Freq']
		std_freq = input_sockets['σ Freq']
		max_e = input_sockets['max E']
		offset = input_sockets['Offset Time']

		has_mean_freq = not ct.FlowSignal.check(mean_freq)
		has_std_freq = not ct.FlowSignal.check(std_freq)
		has_max_e = not ct.FlowSignal.check(max_e)
		has_offset = not ct.FlowSignal.check(offset)

		if has_mean_freq and has_std_freq and has_max_e and has_offset:
			common_params = max_e | mean_freq | std_freq | offset
			match props['active_socket_set']:
				case 'Pulse' | 'Constant':
					return common_params

				case 'Symbolic':
					t_range = input_sockets['t Range']
					has_t_range = not ct.FlowSignal.check(t_range)

					if has_t_range:
						return common_params | t_range | t_range
		return ct.FlowSignal.FlowPending


####################
# - Blender Registration
####################
BL_REGISTER = [
	TemporalShapeNode,
]
BL_NODES = {ct.NodeType.TemporalShape: (ct.NodeCategory.MAXWELLSIM_SOURCES)}
