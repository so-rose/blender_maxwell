"""Implements the `WaveTemporalShapeNode`."""

import functools
import typing as typ

import bpy
import sympy as sp
import sympy.physics.units as spu
import tidy3d as td

from blender_maxwell.utils import extra_sympy_units as spux

from .... import contracts as ct
from .... import managed_objs, sockets
from ... import base, events


class WaveTemporalShapeNode(base.MaxwellSimNode):
	node_type = ct.NodeType.WaveTemporalShape
	bl_label = 'Continuous Wave Temporal Shape'

	####################
	# - Sockets
	####################
	input_sockets: typ.ClassVar = {
		'max E': sockets.ExprSocketDef(
			mathtype=spux.MathType.Complex,
			physical_type=spux.PhysicalType.EField,
			default_value=1 + 0j,
		),
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
		'Offset Time': sockets.ExprSocketDef(default_value=5, abs_min=2.5),
	}
	output_sockets: typ.ClassVar = {
		'Temporal Shape': sockets.MaxwellTemporalShapeSocketDef(),
		'E(t)': sockets.ExprSocketDef(active_kind=ct.FlowKind.Array),
	}

	managed_obj_types: typ.ClassVar = {
		'plot': managed_objs.ManagedBLImage,
	}

	####################
	# - UI
	####################
	def draw_info(self, _: bpy.types.Context, layout: bpy.types.UILayout) -> None:
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
		input_sockets={
			'max E',
			'μ Freq',
			'σ Freq',
			'Offset Time',
		},
		unit_systems={'Tidy3DUnits': ct.UNITS_TIDY3D},
		scale_input_sockets={
			'max E': 'Tidy3DUnits',
			'μ Freq': 'Tidy3DUnits',
			'σ Freq': 'Tidy3DUnits',
		},
	)
	def compute_temporal_shape(self, input_sockets, unit_systems) -> td.GaussianPulse:
		return td.ContinuousWave(
			amplitude=sp.re(input_sockets['max E']),
			phase=sp.im(input_sockets['max E']),
			freq0=input_sockets['μ Freq'],
			fwidth=input_sockets['σ Freq'],
			offset=input_sockets['Offset Time'],
		)

	####################
	# - FlowKind: LazyValueFunc / Info / Params
	####################
	@events.computes_output_socket(
		'E(t)',
		kind=ct.FlowKind.LazyValueFunc,
		output_sockets={'Temporal Shape'},
	)
	def compute_time_to_efield_lazy(self, output_sockets) -> td.GaussianPulse:
		temporal_shape = output_sockets['Temporal Shape']
		jax_amp_time = functools.partial(ct.manual_amp_time, temporal_shape)

		## TODO: Don't just partial() it up, do it property in the ParamsFlow!
		## -> Right now it's recompiled every time.

		return ct.LazyValueFuncFlow(
			func=jax_amp_time,
			func_args=[spux.PhysicalType.Time],
			supports_jax=True,
		)

	@events.computes_output_socket(
		'E(t)',
		kind=ct.FlowKind.Info,
	)
	def compute_time_to_efield_info(self) -> td.GaussianPulse:
		return ct.InfoFlow(
			dim_names=['t'],
			dim_idx={
				't': ct.LazyArrayRangeFlow(
					start=sp.S(0), stop=sp.oo, steps=0, unit=spu.second
				)
			},
			output_name='E',
			output_shape=None,
			output_mathtype=spux.MathType.Complex,
			output_unit=spu.volt / spu.um,
		)

	@events.computes_output_socket(
		'E(t)',
		kind=ct.FlowKind.Params,
	)
	def compute_time_to_efield_params(self) -> td.GaussianPulse:
		sym_time = sp.Symbol('t', real=True, nonnegative=True)
		return ct.ParamsFlow(func_args=[sym_time], symbols={sym_time})


####################
# - Blender Registration
####################
BL_REGISTER = [
	WaveTemporalShapeNode,
]
BL_NODES = {
	ct.NodeType.WaveTemporalShape: (ct.NodeCategory.MAXWELLSIM_SOURCES_TEMPORALSHAPES)
}
