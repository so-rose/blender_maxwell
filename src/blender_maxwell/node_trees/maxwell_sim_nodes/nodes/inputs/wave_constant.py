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
				default_unit=spu.nm,
				default_value=500,
				default_min=200,
				default_max=700,
				default_steps=50,
			)
		},
		'Frequency': {
			'Freq': sockets.ExprSocketDef(
				default_unit=spux.THz,
				default_value=1,
				default_min=0.3,
				default_max=3,
				default_steps=50,
			),
		},
	}
	output_sockets: typ.ClassVar = {
		'WL': sockets.ExprSocketDef(
			physical_type=spux.PhysicalType.Length,
		),
		'Freq': sockets.ExprSocketDef(
			physical_type=spux.PhysicalType.Freq,
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
		col.prop(self, self.blfields['use_range'], toggle=True, text='Range')

	####################
	# - Events
	####################
	@events.on_value_changed(
		prop_name={'active_socket_set', 'use_range'},
		props={'use_range'},
		run_on_init=True,
	)
	def on_use_range_changed(self, props: dict) -> None:
		"""Synchronize the `active_kind` of input/output sockets, to either produce a `ct.FlowKind.Value` or a `ct.FlowKind.Range`."""
		if self.inputs.get('WL') is not None:
			active_input = self.inputs['WL']
		else:
			active_input = self.inputs['Freq']

		# Modify Active Kind(s)
		## Input active_kind -> Value/Range
		active_input_uses_range = active_input.active_kind == ct.FlowKind.Range
		if active_input_uses_range != props['use_range']:
			active_input.active_kind = (
				ct.FlowKind.Range if props['use_range'] else ct.FlowKind.Value
			)

		## Output active_kind -> Value/Range
		for active_output in self.outputs.values():
			active_output_uses_range = active_output.active_kind == ct.FlowKind.Range
			if active_output_uses_range != props['use_range']:
				active_output.active_kind = (
					ct.FlowKind.Range if props['use_range'] else ct.FlowKind.Value
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
		has_wl = not ct.FlowSignal.check(input_sockets['WL'])
		if has_wl:
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
		has_freq = not ct.FlowSignal.check(input_sockets['Freq'])
		if has_freq:
			return input_sockets['Freq']

		return spu.convert_to(
			sci_constants.vac_speed_of_light / input_sockets['WL'], spux.THz
		)

	@events.computes_output_socket(
		'WL',
		kind=ct.FlowKind.Range,
		input_sockets={'WL', 'Freq'},
		input_socket_kinds={
			'WL': ct.FlowKind.Range,
			'Freq': ct.FlowKind.Range,
		},
		input_sockets_optional={'WL': True, 'Freq': True},
	)
	def compute_wl_range(self, input_sockets: dict) -> sp.Expr:
		"""Compute wavelength range from either wavelength/frequency ranges."""
		has_wl = not ct.FlowSignal.check(input_sockets['WL'])
		if has_wl:
			return input_sockets['WL']

		freq = input_sockets['Freq']
		return ct.RangeFlow(
			start=spux.scale_to_unit(
				sci_constants.vac_speed_of_light / (freq.stop * freq.unit), spu.um
			),
			stop=spux.scale_to_unit(
				sci_constants.vac_speed_of_light / (freq.start * freq.unit), spu.um
			),
			steps=freq.steps,
			scaling=freq.scaling,
			unit=spu.um,
		)

	@events.computes_output_socket(
		'Freq',
		kind=ct.FlowKind.Range,
		input_sockets={'WL', 'Freq'},
		input_socket_kinds={
			'WL': ct.FlowKind.Range,
			'Freq': ct.FlowKind.Range,
		},
		input_sockets_optional={'WL': True, 'Freq': True},
	)
	def compute_freq_range(self, input_sockets: dict) -> sp.Expr:
		"""Compute frequency range from either wavelength/frequency ranges."""
		has_freq = not ct.FlowSignal.check(input_sockets['Freq'])
		if has_freq:
			return input_sockets['Freq']

		wl = input_sockets['WL']
		return ct.RangeFlow(
			start=spux.scale_to_unit(
				sci_constants.vac_speed_of_light / (wl.stop * wl.unit), spux.THz
			),
			stop=spux.scale_to_unit(
				sci_constants.vac_speed_of_light / (wl.start * wl.unit), spux.THz
			),
			steps=wl.steps,
			scaling=wl.scaling,
			unit=spux.THz,
		)


####################
# - Blender Registration
####################
BL_REGISTER = [
	WaveConstantNode,
]
BL_NODES = {ct.NodeType.WaveConstant: (ct.NodeCategory.MAXWELLSIM_INPUTS)}
