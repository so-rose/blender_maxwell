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

import functools
import typing as typ

import bpy
import sympy as sp

from blender_maxwell.utils import bl_cache

from ... import contracts as ct
from ... import sockets
from .. import base, events


class CombineNode(base.MaxwellSimNode):
	"""Combine single objects (ex. Source, Monitor, Structure) into a list."""

	node_type = ct.NodeType.Combine
	bl_label = 'Combine'

	####################
	# - Sockets
	####################
	input_socket_sets: typ.ClassVar = {
		'Sources': {},
		'Structures': {},
		'Monitors': {},
	}
	output_socket_sets: typ.ClassVar = {
		'Sources': {
			'Sources': sockets.MaxwellSourceSocketDef(
				active_kind=ct.FlowKind.Array,
			),
		},
		'Structures': {
			'Structures': sockets.MaxwellStructureSocketDef(
				active_kind=ct.FlowKind.Array,
			),
		},
		'Monitors': {
			'Monitors': sockets.MaxwellMonitorSocketDef(
				active_kind=ct.FlowKind.Array,
			),
		},
	}

	####################
	# - Properties
	####################
	concatenate_first: bool = bl_cache.BLField(False)
	value_or_func: ct.FlowKind = bl_cache.BLField(
		enum_cb=lambda self, _: self._value_or_func(),
	)

	def _value_or_func(self):
		return [
			flow_kind.bl_enum_element(i)
			for i, flow_kind in enumerate([ct.FlowKind.Value, ct.FlowKind.Func])
		]

	####################
	# - Draw
	####################
	def draw_props(self, _, layout: bpy.types.UILayout):
		layout.prop(self, self.blfields['value_or_func'], text='')

		if self.value_or_func is ct.FlowKind.Value:
			layout.prop(
				self,
				self.blfields['concatenate_first'],
				text='Concatenate',
				toggle=True,
			)

	####################
	# - Events
	####################
	@events.on_value_changed(
		any_loose_input_socket=True,
		prop_name={'active_socket_set', 'concatenate_first', 'value_or_func'},
		run_on_init=True,
		# Loaded
		props={'active_socket_set', 'concatenate_first', 'value_or_func'},
	)
	def on_inputs_changed(self, props) -> None:
		"""Always create one extra loose input socket."""
		active_socket_set = props['active_socket_set']

		# Deduce SocketDef
		## -> Cheat by retrieving the class from the output sockets.
		SocketDef = self.output_socket_sets[active_socket_set][
			active_socket_set
		].__class__

		# Deduce Current "Filled"
		## -> The first linked socket from the end bounds the "filled" region.
		## -> The length of that region, plus one, will be the new amount.
		reverse_linked_idxs = [
			i
			for i, bl_socket in enumerate(reversed(self.inputs.values()))
			if bl_socket.is_linked
		]
		current_filled = len(self.inputs) - (
			reverse_linked_idxs[0] if reverse_linked_idxs else len(self.inputs)
		)
		new_amount = current_filled + 1

		# Deduce SocketDef | Current Amount
		concatenate_first = props['concatenate_first']
		flow_kind = props['value_or_func']

		self.loose_input_sockets = {
			'#0': SocketDef(
				active_kind=flow_kind
				if flow_kind is ct.FlowKind.Func or not concatenate_first
				else ct.FlowKind.Array
			)
		} | {f'#{i}': SocketDef(active_kind=flow_kind) for i in range(1, new_amount)}

	####################
	# - FlowKind.Array|Func
	####################
	def compute_combined(
		self,
		loose_input_sockets,
		input_flow_kind: typ.Literal[ct.FlowKind.Value, ct.FlowKind.Func],
		output_flow_kind: typ.Literal[ct.FlowKind.Array, ct.FlowKind.Func],
	) -> list[typ.Any] | ct.FuncFlow | ct.FlowSignal:
		"""Correctly compute the combined loose input sockets, given a valid combination of input and output `FlowKind`s.

		If there is no output, or the flows aren't compatible, return `FlowPending`.
		"""
		match (input_flow_kind, output_flow_kind):
			case (ct.FlowKind.Value, ct.FlowKind.Array):
				value_flows = [
					inp
					for inp in loose_input_sockets.values()
					if not ct.FlowSignal.check(inp)
				]
				if value_flows:
					return value_flows
				return ct.FlowSignal.FlowPending

			case (ct.FlowKind.Func, ct.FlowKind.Func):
				func_flows = [
					inp
					for inp in loose_input_sockets.values()
					if not ct.FlowSignal.check(inp)
				]
				if func_flows:
					return functools.reduce(
						lambda a, b: a | b,
						func_flows,
					)
				return ct.FlowSignal.FlowPending

		return ct.FlowSignal.FlowPending

	####################
	# - Output: Sources
	####################
	@events.computes_output_socket(
		'Sources',
		kind=ct.FlowKind.Array,
		all_loose_input_sockets=True,
		props={'value_or_func'},
	)
	def compute_sources_array(
		self, props, loose_input_sockets
	) -> list[typ.Any] | ct.FlowSignal:
		"""Compute sources."""
		return self.compute_combined(
			loose_input_sockets, props['value_or_func'], ct.FlowKind.Array
		)

	@events.computes_output_socket(
		'Sources',
		kind=ct.FlowKind.Func,
		all_loose_input_sockets=True,
		props={'value_or_func'},
	)
	def compute_sources_func(self, props, loose_input_sockets) -> list[typ.Any]:
		"""Compute (lazy) sources."""
		return self.compute_combined(
			loose_input_sockets, props['value_or_func'], ct.FlowKind.Func
		)

	####################
	# - Output: Structures
	####################
	@events.computes_output_socket(
		'Structures',
		kind=ct.FlowKind.Array,
		all_loose_input_sockets=True,
		props={'value_or_func'},
	)
	def compute_structures_array(self, props, loose_input_sockets) -> sp.Expr:
		"""Compute structures."""
		return self.compute_combined(
			loose_input_sockets, props['value_or_func'], ct.FlowKind.Array
		)

	@events.computes_output_socket(
		'Structures',
		kind=ct.FlowKind.Func,
		all_loose_input_sockets=True,
		props={'value_or_func'},
	)
	def compute_structures_func(self, props, loose_input_sockets) -> list[typ.Any]:
		"""Compute (lazy) structures."""
		return self.compute_combined(
			loose_input_sockets, props['value_or_func'], ct.FlowKind.Func
		)

	####################
	# - Output: Monitors
	####################
	@events.computes_output_socket(
		'Monitors',
		kind=ct.FlowKind.Array,
		all_loose_input_sockets=True,
		props={'value_or_func'},
	)
	def compute_monitors_array(self, props, loose_input_sockets) -> sp.Expr:
		"""Compute monitors."""
		return self.compute_combined(
			loose_input_sockets, props['value_or_func'], ct.FlowKind.Array
		)

	@events.computes_output_socket(
		'Monitors',
		kind=ct.FlowKind.Func,
		all_loose_input_sockets=True,
		props={'value_or_func'},
	)
	def compute_monitors_func(self, props, loose_input_sockets) -> list[typ.Any]:
		"""Compute (lazy) monitors."""
		return self.compute_combined(
			loose_input_sockets, props['value_or_func'], ct.FlowKind.Func
		)


####################
# - Blender Registration
####################
BL_REGISTER = [
	CombineNode,
]
BL_NODES = {ct.NodeType.Combine: (ct.NodeCategory.MAXWELLSIM_SIMS)}
