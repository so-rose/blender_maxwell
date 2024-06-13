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

from blender_maxwell.utils import bl_cache, logger, sci_constants, sim_symbols
from blender_maxwell.utils import sympy_extra as spux

from ... import contracts as ct
from ... import sockets
from .. import base, events

log = logger.get(__name__)

FK = ct.FlowKind
FS = ct.FlowSignal
MT = spux.MathType


class WaveConstantNode(base.MaxwellSimNode):
	"""Translates vacuum wavelength / non-angular frequency into both, either as a scalar, or as a memory-efficient uniform range of values.

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
		"""Synchronize the `active_kind` of input/output sockets, to either produce a `FK.Value` or a `FK.Range`."""
		if self.inputs.get('WL') is not None:
			active_input = self.inputs['WL']
		else:
			active_input = self.inputs['Freq']

		# Modify Active Kind(s)
		## Input active_kind -> Value/Range
		active_input_uses_range = active_input.active_kind == FK.Range
		if active_input_uses_range != props['use_range']:
			active_input.active_kind = FK.Range if props['use_range'] else FK.Value

		## Output active_kind -> Value/Range
		for active_output in self.outputs.values():
			active_output_uses_range = active_output.active_kind == FK.Range
			if active_output_uses_range != props['use_range']:
				active_output.active_kind = FK.Range if props['use_range'] else FK.Value

	####################
	# - FlowKind.Value
	####################
	@events.computes_output_socket(
		'WL',
		kind=FK.Value,
		# Loaded
		inscks_kinds={'WL': FK.Value, 'Freq': FK.Value},
		input_sockets_optional={'WL', 'Freq'},
	)
	def compute_wl_value(self, input_sockets: dict) -> sp.Expr:
		"""Compute a single wavelength value from either wavelength/frequency."""
		has_wl = not FS.check(input_sockets['WL'])
		if has_wl:
			return input_sockets['WL']

		return spu.convert_to(
			sci_constants.vac_speed_of_light / input_sockets['Freq'], spu.um
		)

	@events.computes_output_socket(
		'Freq',
		kind=FK.Value,
		# Loaded
		inscks_kinds={'WL': FK.Value, 'Freq': FK.Value},
		input_sockets_optional={'WL', 'Freq'},
	)
	def compute_freq_value(self, input_sockets: dict) -> sp.Expr:
		"""Compute a single frequency value from either wavelength/frequency."""
		has_freq = not FS.check(input_sockets['Freq'])
		if has_freq:
			return input_sockets['Freq']

		return spu.convert_to(
			sci_constants.vac_speed_of_light / input_sockets['WL'], spux.THz
		)

	####################
	# - FlowKind.Range
	####################
	@events.computes_output_socket(
		'WL',
		kind=FK.Range,
		# Loaded
		inscks_kinds={'WL': FK.Range, 'Freq': FK.Range},
		input_sockets_optional={'WL', 'Freq'},
	)
	def compute_wl_range(self, input_sockets) -> sp.Expr:
		"""Compute wavelength range from either wavelength/frequency ranges."""
		has_wl = not FS.check(input_sockets['WL'])
		if has_wl:
			return input_sockets['WL']

		freq = input_sockets['Freq']
		return freq.rescale(
			lambda bound: sci_constants.vac_speed_of_light / bound,
			reverse=True,
			new_unit=spu.um,
		)

	@events.computes_output_socket(
		'Freq',
		kind=FK.Range,
		input_sockets={'WL', 'Freq'},
		input_socket_kinds={
			'WL': FK.Range,
			'Freq': FK.Range,
		},
		input_sockets_optional={'WL': True, 'Freq': True},
	)
	def compute_freq_range(self, input_sockets: dict) -> sp.Expr:
		"""Compute frequency range from either wavelength/frequency ranges."""
		has_freq = not FS.check(input_sockets['Freq'])
		if has_freq:
			return input_sockets['Freq']

		wl = input_sockets['WL']
		return wl.rescale(
			lambda bound: sci_constants.vac_speed_of_light / bound,
			reverse=True,
			new_unit=spux.THz,
		)

	####################
	# - FlowKind.Func
	####################
	# @events.computes_output_socket(
	# 'WL',
	# kind=FK.Func,
	# # Loaded
	# inscks_kinds={'WL': FK.Func, 'Freq': FK.Func},
	# input_sockets_optional={'WL', 'Freq'},
	# )
	# def compute_wl_func(self, input_sockets: dict) -> ct.FuncFlow | FS:
	# """Compute a single wavelength value from either wavelength/frequency."""
	# wl = input_sockets['WL']
	# has_wl = not FS.check(wl)
	# if has_wl:
	# return wl

	# freq = input_sockets['Freq']
	# has_freq = not FS.check(freq)
	# if has_freq:
	# return wl.compose_within(
	# return spu.convert_to(
	# sci_constants.vac_speed_of_light / input_sockets['Freq'], spu.um
	# )

	# return FS.FlowPending

	# @events.computes_output_socket(
	# 'Freq',
	# kind=FK.Value,
	# # Loaded
	# inscks_kinds={'WL': FK.Value, 'Freq': FK.Value},
	# input_sockets_optional={'WL', 'Freq'},
	# )
	# def compute_freq_value(self, input_sockets: dict) -> sp.Expr:
	# """Compute a single frequency value from either wavelength/frequency."""
	# has_freq = not FS.check(input_sockets['Freq'])
	# if has_freq:
	# return input_sockets['Freq']

	# return spu.convert_to(
	# sci_constants.vac_speed_of_light / input_sockets['WL'], spux.THz
	# )

	####################
	# - FlowKind.Info
	####################
	@events.computes_output_socket(
		'WL',
		kind=FK.Info,
	)
	def compute_wl_info(self) -> ct.InfoFlow:
		"""Just enough InfoFlow to enable `linked_capabilities`."""
		return ct.InfoFlow(
			output=sim_symbols.wl(spu.um),
		)

	@events.computes_output_socket(
		'Freq',
		kind=FK.Info,
	)
	def compute_freq_info(self) -> sp.Expr:
		"""Compute frequency range from either wavelength/frequency ranges."""
		return ct.InfoFlow(
			output=sim_symbols.freq(spux.THz),
		)


####################
# - Blender Registration
####################
BL_REGISTER = [
	WaveConstantNode,
]
BL_NODES = {ct.NodeType.WaveConstant: (ct.NodeCategory.MAXWELLSIM_UTILITIES)}
