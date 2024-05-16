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

import typing as typ

import bpy
import sympy as sp
import sympy.physics.units as spu
import tidy3d as td

from blender_maxwell.utils import extra_sympy_units as spux
from blender_maxwell.utils import logger, sim_symbols

from ... import contracts as ct
from ... import managed_objs, sockets
from .. import base, events

log = logger.get(__name__)


_max_e_socket_def = sockets.ExprSocketDef(
	mathtype=spux.MathType.Complex,
	physical_type=spux.PhysicalType.EField,
	default_value=1 + 0j,
)
_offset_socket_def = sockets.ExprSocketDef(default_value=5, abs_min=2.5)


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
	}
	input_socket_sets: typ.ClassVar = {
		'Pulse': {
			'max E': _max_e_socket_def,
			'Offset Time': _offset_socket_def,
			'Remove DC': sockets.BoolSocketDef(default_value=True),
		},
		'Constant': {
			'max E': _max_e_socket_def,
			'Offset Time': _offset_socket_def,
		},
		'Symbolic': {
			't Range': sockets.ExprSocketDef(
				active_kind=ct.FlowKind.LazyArrayRange,
				physical_type=spux.PhysicalType.Time,
				default_unit=spu.picosecond,
				default_min=0,
				default_max=10,
				default_steps=100,
			),
			'Envelope': sockets.ExprSocketDef(
				default_symbols=[sim_symbols.t_ps],
				default_value=10 * sim_symbols.t_ps.sp_symbol,
			),
		},
	}

	output_sockets: typ.ClassVar = {
		'Temporal Shape': sockets.MaxwellTemporalShapeSocketDef(),
	}

	managed_obj_types: typ.ClassVar = {
		'plot': managed_objs.ManagedBLImage,
	}

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
	# - FlowKind: Value
	####################
	@events.computes_output_socket(
		'Temporal Shape',
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
			't Range': ct.FlowKind.LazyArrayRange,
			'Envelope': ct.FlowKind.LazyValueFunc,
		},
		input_sockets_optional={
			'max E': True,
			'Offset Time': True,
			'Remove DC': True,
			't Range': True,
			'Envelope': True,
		},
		unit_systems={'Tidy3DUnits': ct.UNITS_TIDY3D},
		scale_input_sockets={
			'max E': 'Tidy3DUnits',
			'μ Freq': 'Tidy3DUnits',
			'σ Freq': 'Tidy3DUnits',
			't Range': 'Tidy3DUnits',
			'Offset Time': 'Tidy3DUnits',
		},
	)
	def compute_temporal_shape(
		self, props, input_sockets, unit_systems
	) -> td.GaussianPulse:
		match props['active_socket_set']:
			case 'Pulse':
				return td.GaussianPulse(
					amplitude=sp.re(input_sockets['max E']),
					phase=sp.im(input_sockets['max E']),
					freq0=input_sockets['μ Freq'],
					fwidth=input_sockets['σ Freq'],
					offset=input_sockets['Offset Time'],
					remove_dc_component=input_sockets['Remove DC'],
				)

			case 'Constant':
				return td.ContinuousWave(
					amplitude=sp.re(input_sockets['max E']),
					phase=sp.im(input_sockets['max E']),
					freq0=input_sockets['μ Freq'],
					fwidth=input_sockets['σ Freq'],
					offset=input_sockets['Offset Time'],
				)

			case 'Symbolic':
				lzrange = input_sockets['t Range']
				envelope_ps = input_sockets['Envelope'].func_jax

				return td.CustomSourceTime.from_values(
					freq0=input_sockets['μ Freq'],
					fwidth=input_sockets['σ Freq'],
					values=envelope_ps(
						lzrange.rescale_to_unit(spu.ps).realize_array.values
					),
					dt=input_sockets['t Range'].realize_step_size(),
				)


####################
# - Blender Registration
####################
BL_REGISTER = [
	TemporalShapeNode,
]
BL_NODES = {ct.NodeType.TemporalShape: (ct.NodeCategory.MAXWELLSIM_SOURCES)}
