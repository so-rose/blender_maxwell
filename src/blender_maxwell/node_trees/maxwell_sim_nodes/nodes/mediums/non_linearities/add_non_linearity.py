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

"""Implements `AddNonLinearity`."""

import functools
import typing as typ

import tidy3d as td

from blender_maxwell.utils import logger
from blender_maxwell.utils import sympy_extra as spux

from .... import contracts as ct
from .... import sockets
from ... import base, events

log = logger.get(__name__)

FK = ct.FlowKind
FS = ct.FlowSignal
MT = spux.MathType


class AddNonLinearity(base.MaxwellSimNode):
	"""Add non-linearities to a medium, increasing the range of effects that it can encapsulate."""

	node_type = ct.NodeType.AddNonLinearity
	bl_label = 'Add Non-Linearity'

	input_sockets: typ.ClassVar = {
		'Medium': sockets.MaxwellMediumSocketDef(active_kind=FK.Func),
		'Iterations': sockets.ExprSocketDef(
			mathtype=MT.Integer,
			default_value=5,
			abs_min=1,
		),
	}
	output_sockets: typ.ClassVar = {
		'Medium': sockets.MaxwellMediumSocketDef(active_kind=FK.Func),
	}

	####################
	# - Events
	####################
	@events.on_value_changed(
		any_loose_input_socket=True,
		run_on_init=True,
	)
	def on_inputs_changed(self) -> None:
		"""Always create one extra loose input socket off the end of the last linked loose socket."""
		# Deduce SocketDef
		## -> Cheat by retrieving the class from the output sockets.
		SocketDef = sockets.MaxwellMediumNonLinearitySocketDef

		## TODO: Move this code to events, so it can be shared w/Combine
		# Deduce Current "Filled"
		## -> The first linked socket from the end bounds the "filled" region.
		## -> The length of that region, plus one, will be the new amount.
		total_loose_inputs = len(self.loose_input_sockets)

		reverse_linked_idxs = [
			i
			for i, bl_socket in enumerate(reversed(self.inputs.values()))
			if i < total_loose_inputs and bl_socket.is_linked
		]
		current_filled = total_loose_inputs - (
			reverse_linked_idxs[0] if reverse_linked_idxs else total_loose_inputs
		)
		new_amount = current_filled + 1

		# Deduce SocketDef | Current Amount
		self.loose_input_sockets = {
			'#0': SocketDef(active_kind=FK.Func),
		} | {f'#{i}': SocketDef(active_kind=FK.Func) for i in range(1, new_amount)}

	####################
	# - FlowKind.Value
	####################
	@events.computes_output_socket(
		'Medium',
		kind=FK.Value,
		# Loaded
		outscks_kinds={
			'Medium': {FK.Func, FK.Params},
		},
	)
	def compute_value(self, output_sockets) -> ct.ParamsFlow | FS:
		"""Compute the particular value of the simulation domain from strictly non-symbolic inputs."""
		value = events.realize_known(output_sockets['Medium'])
		if value is not None:
			return value
		return ct.FlowSignal.FlowPending

	####################
	# - FlowKind.Func
	####################
	@events.computes_output_socket(
		'Medium',
		kind=FK.Func,
		# Loaded
		inscks_kinds={
			'Medium': FK.Func,
			'Iterations': FK.Func,
		},
		all_loose_input_sockets=True,
		loose_input_sockets_kind=FK.Func,
	)
	def compute_func(self, input_sockets, loose_input_sockets) -> ct.ParamsFlow | FS:
		"""Compute the particular value of the simulation domain from strictly non-symbolic inputs."""
		medium = input_sockets['Medium']
		iterations = input_sockets['Iterations']

		funcs = [
			non_linearity
			for non_linearity in loose_input_sockets.values()
			if not FS.check(non_linearity)
		]

		if funcs:
			non_linearities = functools.reduce(
				lambda a, b: a | b,
				funcs,
			)

			return (medium | iterations | non_linearities).compose_within(
				lambda els: els[0].updated_copy(
					nonlinear_spec=td.NonlinearSpec(
						num_iters=els[1],
						models=els[2] if isinstance(els[2], tuple) else [els[2]],
					)
				)
			)
		return ct.FlowSignal.FlowPending

	####################
	# - FlowKind.Params
	####################
	@events.computes_output_socket(
		'Medium',
		kind=ct.FlowKind.Params,
		# Loaded
		inscks_kinds={
			'Medium': FK.Params,
			'Iterations': FK.Params,
		},
		all_loose_input_sockets=True,
		loose_input_sockets_kind=FK.Params,
	)
	def compute_params(self, input_sockets, loose_input_sockets) -> td.Box:
		"""Aggregate the function parameters needed by the box."""
		medium = input_sockets['Medium']
		iterations = input_sockets['Iterations']

		funcs = [
			non_linearity
			for non_linearity in loose_input_sockets.values()
			if not FS.check(non_linearity)
		]

		if funcs:
			non_linearities = functools.reduce(
				lambda a, b: a | b,
				funcs,
			)

			return medium | iterations | non_linearities
		return ct.FlowSignal.FlowPending


####################
# - Blender Registration
####################
BL_REGISTER = [
	AddNonLinearity,
]
BL_NODES = {
	ct.NodeType.AddNonLinearity: (ct.NodeCategory.MAXWELLSIM_MEDIUMS_NONLINEARITIES)
}
