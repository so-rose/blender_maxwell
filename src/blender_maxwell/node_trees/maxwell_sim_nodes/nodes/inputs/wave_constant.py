"""Implements `WaveConstantNode`."""

import typing as typ

import bpy
import sympy as sp
import sympy.physics.units as spu

from blender_maxwell.utils import bl_cache, logger, sci_constants
from blender_maxwell.utils import extra_sympy_units as spux

from ... import contracts as ct
from ... import sockets
from .. import base, events

log = logger.get(__name__)


class WaveConstantNode(base.MaxwellSimNode):
	"""Translates vaccum wavelength/frequency into both, either as a scalar, or as a memory-efficient uniform range of values.

	Socket Sets:
		Wavelength: Input a wavelength (range) to produce both wavelength/frequency (ranges).
		Frequency: Input a frequency (range) to produce both wavelength/frequency (ranges).

	Attributes:
		use_range: Whether to specify a range of wavelengths/frequencies, or just one.
	"""

	node_type = ct.NodeType.WaveConstant
	bl_label = 'Wave Constant'

	input_socket_sets: typ.ClassVar = {
		'Wavelength': {
			'WL': sockets.ExprSocketDef(
				active_kind=ct.FlowKind.Value,
				unit_dimension=spux.unit_dims.length,
				# Defaults
				default_unit=spu.nm,
				default_value=500,
				default_min=200,
				default_max=700,
				default_steps=2,
			)
		},
		'Frequency': {
			'Freq': sockets.ExprSocketDef(
				active_kind=ct.FlowKind.Value,
				unit_dimension=spux.unit_dims.frequency,
				# Defaults
				default_unit=spux.THz,
				default_value=1,
				default_min=0.3,
				default_max=3,
				default_steps=2,
			),
		},
	}
	output_sockets: typ.ClassVar = {
		'WL': sockets.ExprSocketDef(
			active_kind=ct.FlowKind.Value,
			unit_dimension=spux.unit_dims.length,
		),
		'Freq': sockets.ExprSocketDef(
			active_kind=ct.FlowKind.Value,
			unit_dimension=spux.unit_dims.frequency,
		),
	}

	####################
	# - Properties
	####################
	use_range: bool = bl_cache.BLField(False)

	####################
	# - UI
	####################
	def draw_props(self, _: bpy.types.Context, col: bpy.types.UILayout) -> None:
		"""Draws the button that allows toggling between single and range output.

		Parameters:
			col: Target for defining UI elements.
		"""
		col.prop(self, self.blfields['use_range'], toggle=True)

	####################
	# - Events
	####################
	@events.on_value_changed(
		prop_name={'active_socket_set', 'use_range'},
		props='use_range',
		run_on_init=True,
	)
	def on_use_range_changed(self, props: dict) -> None:
		"""Synchronize the `active_kind` of input/output sockets, to either produce a `ct.FlowKind.Value` or a `ct.FlowKind.LazyArrayRange`."""
		if self.inputs.get('WL') is not None:
			active_input = self.inputs['WL']
		else:
			active_input = self.inputs['Freq']

		# Modify Active Kind(s)
		## Input active_kind -> Value/LazyArrayRange
		active_input_uses_range = active_input.active_kind == ct.FlowKind.LazyArrayRange
		if active_input_uses_range != props['use_range']:
			active_input.active_kind = (
				ct.FlowKind.LazyArrayRange if props['use_range'] else ct.FlowKind.Value
			)

		## Output active_kind -> Value/LazyArrayRange
		for active_output in self.outputs.values():
			active_output_uses_range = (
				active_output.active_kind == ct.FlowKind.LazyArrayRange
			)
			if active_output_uses_range != props['use_range']:
				active_output.active_kind = (
					ct.FlowKind.LazyArrayRange
					if props['use_range']
					else ct.FlowKind.Value
				)

	####################
	# - FlowKinds
	####################
	@events.computes_output_socket(
		'WL',
		kind=ct.FlowKind.Value,
		input_sockets={'WL', 'Freq'},
		input_sockets_optional={'WL': True, 'Freq': True},
	)
	def compute_wl_value(self, input_sockets: dict) -> sp.Expr:
		"""Compute a single wavelength value from either wavelength/frequency."""
		if input_sockets['WL'] is not None:
			return input_sockets['WL']

		return sci_constants.vac_speed_of_light / input_sockets['Freq']

	@events.computes_output_socket(
		'Freq',
		kind=ct.FlowKind.Value,
		input_sockets={'WL', 'Freq'},
		input_sockets_optional={'WL': True, 'Freq': True},
	)
	def compute_freq_value(self, input_sockets: dict) -> sp.Expr:
		"""Compute a single frequency value from either wavelength/frequency."""
		if input_sockets['Freq'] is not None:
			return input_sockets['Freq']

		return sci_constants.vac_speed_of_light / input_sockets['WL']

	@events.computes_output_socket(
		'WL',
		kind=ct.FlowKind.LazyArrayRange,
		input_sockets={'WL', 'Freq'},
		input_socket_kinds={
			'WL': ct.FlowKind.LazyArrayRange,
			'Freq': ct.FlowKind.LazyArrayRange,
		},
		input_sockets_optional={'WL': True, 'Freq': True},
	)
	def compute_wl_range(self, input_sockets: dict) -> sp.Expr:
		"""Compute wavelength range from either wavelength/frequency ranges."""
		if input_sockets['WL'] is not None:
			return input_sockets['WL']

		return input_sockets['Freq'].rescale_bounds(
			lambda bound: sci_constants.vac_speed_of_light / bound, reverse=True
		)

	@events.computes_output_socket(
		'Freq',
		kind=ct.FlowKind.LazyArrayRange,
		input_sockets={'WL', 'Freq'},
		input_socket_kinds={
			'WL': ct.FlowKind.LazyArrayRange,
			'Freq': ct.FlowKind.LazyArrayRange,
		},
		input_sockets_optional={'WL': True, 'Freq': True},
	)
	def compute_freq_range(self, input_sockets: dict) -> sp.Expr:
		"""Compute frequency range from either wavelength/frequency ranges."""
		if input_sockets['Freq'] is not None:
			return input_sockets['Freq']

		return input_sockets['WL'].rescale_bounds(
			lambda bound: sci_constants.vac_speed_of_light / bound, reverse=True
		)


####################
# - Blender Registration
####################
BL_REGISTER = [
	WaveConstantNode,
]
BL_NODES = {ct.NodeType.WaveConstant: (ct.NodeCategory.MAXWELLSIM_INPUTS)}
