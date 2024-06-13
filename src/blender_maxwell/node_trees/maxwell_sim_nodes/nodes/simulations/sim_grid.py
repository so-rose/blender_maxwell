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

"""Implements `SimGridNode`."""

import typing as typ

import sympy.physics.units as spu
import tidy3d as td

from blender_maxwell.utils import logger

from ... import contracts as ct
from ... import sockets
from .. import base, events

log = logger.get(__name__)


SSA = ct.SimSpaceAxis
FK = ct.FlowKind
FS = ct.FlowSignal


class SimGridNode(base.MaxwellSimNode):
	"""Provides a hub for joining custom simulation domain boundary conditions by-axis."""

	node_type = ct.NodeType.SimGrid
	bl_label = 'Sim Grid'

	####################
	# - Sockets
	####################
	input_sockets: typ.ClassVar = {
		'X': sockets.MaxwellSimGridAxisSocketDef(active_kind=FK.Func),
		'Y': sockets.MaxwellSimGridAxisSocketDef(active_kind=FK.Func),
		'Z': sockets.MaxwellSimGridAxisSocketDef(active_kind=FK.Func),
	}
	input_socket_sets: typ.ClassVar = {
		'Relative': {},
		'Absolute': {
			'WL': sockets.ExprSocketDef(
				default_unit=spu.nm,
				default_value=500,
				abs_min=0,
				abs_min_closed=False,
			)
		},
	}
	output_sockets: typ.ClassVar = {
		'Grid': sockets.MaxwellSimGridSocketDef(),
	}

	####################
	# - FlowKind.Value
	####################
	@events.computes_output_socket(
		'Grid',
		kind=FK.Value,
		# Loaded
		outscks_kinds={'Grid': {FK.Func, FK.Params}},
	)
	def compute_bcs_value(self, output_sockets) -> td.BoundarySpec:
		"""Compute the simulation boundary conditions, by combining the individual input by specified half axis."""
		value = events.realize_known(output_sockets['Grid'])
		if value is not None:
			return value
		return FS.FlowPending

	####################
	# - FlowKind.Func
	####################
	@events.computes_output_socket(
		'Grid',
		kind=FK.Func,
		# Loaded
		props={'active_socket_set'},
		inscks_kinds={
			'X': FK.Func,
			'Y': FK.Func,
			'Z': FK.Func,
			'WL': FK.Func,
		},
		input_sockets_optional={'WL'},
		scale_input_sockets={
			'WL': ct.UNITS_TIDY3D,
		},
	)
	def compute_grid_func(self, props, input_sockets) -> ct.ParamsFlow | FS:
		"""Compute the simulation grid lazily, at the specified wavelength."""
		# Deduce "Doubledness"
		## -> A "doubled" axis defines the same bound cond both ways
		x = input_sockets['X']
		y = input_sockets['Y']
		z = input_sockets['Z']

		wl = input_sockets['WL']

		active_socket_set = props['active_socket_set']
		common_func = x | y | z
		match active_socket_set:
			case 'Absolute' if not FS.check(wl):
				return (common_func | wl).compose_within(
					lambda els: td.GridSpec(
						grid_x=els[0], grid_y=els[1], grid_z=els[2], wavelength=els[3]
					)
				)

			case 'Relative':
				return common_func.compose_within(
					lambda els: td.GridSpec(
						grid_x=els[0],
						grid_y=els[1],
						grid_z=els[2],
					)
				)

		return FS.FlowPending

	####################
	# - FlowKind.Params
	####################
	@events.computes_output_socket(
		'Grid',
		kind=FK.Params,
		# Loaded
		props={'active_socket_set'},
		inscks_kinds={
			'X': FK.Params,
			'Y': FK.Params,
			'Z': FK.Params,
			'WL': FK.Params,
		},
		input_sockets_optional={'WL'},
	)
	def compute_bcs_params(self, props, input_sockets) -> ct.ParamsFlow | FS:
		"""Compute the simulation boundary conditions, by combining the individual input by specified half axis."""
		# Deduce "Doubledness"
		## -> A "doubled" axis defines the same bound cond both ways
		x = input_sockets['X']
		y = input_sockets['Y']
		z = input_sockets['Z']

		wl = input_sockets['WL']

		active_socket_set = props['active_socket_set']
		common_params = x | y | z
		match active_socket_set:
			case 'Relative':
				return common_params

			case 'Absolute' if not FS.check(wl):
				return common_params | wl

		return FS.FlowPending


####################
# - Blender Registration
####################
BL_REGISTER = [
	SimGridNode,
]
BL_NODES = {ct.NodeType.SimGrid: (ct.NodeCategory.MAXWELLSIM_SIMS)}
