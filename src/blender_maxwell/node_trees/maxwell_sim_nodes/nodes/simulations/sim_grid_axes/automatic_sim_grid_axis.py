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

"""Implements `AutoSimGridAxisNode`."""

import typing as typ

import sympy.physics.units as spu
import tidy3d as td

from blender_maxwell.utils import logger

from .... import contracts as ct
from .... import sockets
from ... import base, events

log = logger.get(__name__)


SSA = ct.SimSpaceAxis
FK = ct.FlowKind
FS = ct.FlowSignal


class AutoSimGridAxisNode(base.MaxwellSimNode):
	"""Declare a uniform grid along a simulation axis."""

	node_type = ct.NodeType.AutoSimGridAxis
	bl_label = 'Auto Grid Axis'

	####################
	# - Sockets
	####################
	input_sockets: typ.ClassVar = {
		'min N/λ': sockets.ExprSocketDef(
			default_value=10,
		),
		'min Δℓ': sockets.ExprSocketDef(
			default_unit=spu.nm,
			default_value=0,
			abs_min=0,
		),
		'max ratio': sockets.ExprSocketDef(
			default_value=1.4,
			abs_min=1,
		),
	}
	output_sockets: typ.ClassVar = {
		'Grid Axis': sockets.MaxwellSimGridAxisSocketDef(active_kind=FK.Func),
	}

	####################
	# - FlowKind.Value
	####################
	@events.computes_output_socket(
		'Grid Axis',
		kind=FK.Value,
		# Loaded
		outscks_kinds={'Grid Axis': {FK.Func, FK.Params}},
	)
	def compute_bcs_value(self, output_sockets) -> td.BoundarySpec:
		"""Compute the simulation boundary conditions, by combining the individual input by specified half axis."""
		value = events.realize_known(output_sockets['Grid Axis'])
		if value is not None:
			return value
		return FS.FlowPending

	####################
	# - FlowKind.Func
	####################
	@events.computes_output_socket(
		'Grid Axis',
		kind=FK.Func,
		# Loaded
		inscks_kinds={
			'min N/λ': FK.Func,
			'max ratio': FK.Func,
			'min Δℓ': FK.Func,
		},
		scale_input_sockets={
			'min Δℓ': ct.UNITS_TIDY3D,
		},
	)
	def compute_grid_func(self, input_sockets) -> ct.ParamsFlow | FS:
		"""Compute the simulation grid lazily, at the specified wavelength."""
		min_steps_per_wl = input_sockets['min N/λ']
		max_consecutive_ratio = input_sockets['max ratio']
		min_length = input_sockets['min Δℓ']

		return (min_steps_per_wl | max_consecutive_ratio | min_length).compose_within(
			lambda els: td.AutoGrid(
				min_steps_per_wvl=els[0],
				max_scale=els[1],
				dl_min=els[2],
			)
		)

	####################
	# - FlowKind.Params
	####################
	@events.computes_output_socket(
		'Grid Axis',
		kind=FK.Params,
		# Loaded
		inscks_kinds={
			'min N/λ': FK.Params,
			'max ratio': FK.Params,
			'min Δℓ': FK.Params,
		},
	)
	def compute_grid_params(self, input_sockets) -> ct.ParamsFlow | FS:
		"""Compute the simulation grid lazily, at the specified wavelength."""
		min_steps_per_wl = input_sockets['min N/λ']
		max_consecutive_ratio = input_sockets['max ratio']
		min_length = input_sockets['min Δℓ']

		return min_steps_per_wl | max_consecutive_ratio | min_length


####################
# - Blender Registration
####################
BL_REGISTER = [
	AutoSimGridAxisNode,
]
BL_NODES = {ct.NodeType.AutoSimGridAxis: (ct.NodeCategory.MAXWELLSIM_SIMS_SIMGRIDAXES)}
