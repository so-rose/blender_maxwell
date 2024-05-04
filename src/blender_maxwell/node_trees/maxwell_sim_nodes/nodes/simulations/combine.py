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

import typing as typ

import sympy as sp

from blender_maxwell.utils import bl_cache

from ... import contracts as ct
from ... import sockets
from .. import base, events


class CombineNode(base.MaxwellSimNode):
	node_type = ct.NodeType.Combine
	bl_label = 'Combine'

	####################
	# - Sockets
	####################
	input_socket_sets: typ.ClassVar = {
		'Maxwell Sources': {},
		'Maxwell Structures': {},
		'Maxwell Monitors': {},
	}
	output_socket_sets: typ.ClassVar = {
		'Maxwell Sources': {
			'Sources': sockets.MaxwellSourceSocketDef(
				is_list=True,
			),
		},
		'Maxwell Structures': {
			'Structures': sockets.MaxwellStructureSocketDef(
				is_list=True,
			),
		},
		'Maxwell Monitors': {
			'Monitors': sockets.MaxwellMonitorSocketDef(
				is_list=True,
			),
		},
	}

	####################
	# - Draw
	####################
	amount: int = bl_cache.BLField(2, abs_min=1, prop_ui=True)

	####################
	# - Draw
	####################
	def draw_props(self, context, layout):
		layout.prop(self, self.blfields['amount'], text='')

	####################
	# - Events
	####################
	@events.on_value_changed(
		# Trigger
		prop_name={'active_socket_set', 'amount'},
		props={'active_socket_set', 'amount'},
		run_on_init=True,
	)
	def on_inputs_changed(self, props):
		if props['active_socket_set'] == 'Maxwell Sources':
			if (
				not self.loose_input_sockets
				or not next(iter(self.loose_input_sockets)).startswith('Source')
				or len(self.loose_input_sockets) != props['amount']
			):
				self.loose_input_sockets = {
					f'Source #{i}': sockets.MaxwellSourceSocketDef()
					for i in range(props['amount'])
				}

		elif props['active_socket_set'] == 'Maxwell Structures':
			if (
				not self.loose_input_sockets
				or not next(iter(self.loose_input_sockets)).startswith('Structure')
				or len(self.loose_input_sockets) != props['amount']
			):
				self.loose_input_sockets = {
					f'Structure #{i}': sockets.MaxwellStructureSocketDef()
					for i in range(props['amount'])
				}
		elif props['active_socket_set'] == 'Maxwell Monitors':
			if (
				not self.loose_input_sockets
				or not next(iter(self.loose_input_sockets)).startswith('Monitor')
				or len(self.loose_input_sockets) != props['amount']
			):
				self.loose_input_sockets = {
					f'Monitor #{i}': sockets.MaxwellMonitorSocketDef()
					for i in range(props['amount'])
				}
		elif self.loose_input_sockets:
			self.loose_input_sockets = {}

	####################
	# - Output Socket Computation
	####################
	@events.computes_output_socket(
		'Sources',
		kind=ct.FlowKind.Array,
		all_loose_input_sockets=True,
		props={'amount'},
	)
	def compute_sources(self, loose_input_sockets, props) -> sp.Expr:
		return [loose_input_sockets[f'Source #{i}'] for i in range(props['amount'])]

	@events.computes_output_socket(
		'Structures',
		kind=ct.FlowKind.Array,
		all_loose_input_sockets=True,
		props={'amount'},
	)
	def compute_structures(self, loose_input_sockets, props) -> sp.Expr:
		return [loose_input_sockets[f'Structure #{i}'] for i in range(props['amount'])]

	@events.computes_output_socket(
		'Monitors',
		kind=ct.FlowKind.Array,
		all_loose_input_sockets=True,
		props={'amount'},
	)
	def compute_monitors(self, loose_input_sockets, props) -> sp.Expr:
		return [loose_input_sockets[f'Monitor #{i}'] for i in range(props['amount'])]


####################
# - Blender Registration
####################
BL_REGISTER = [
	CombineNode,
]
BL_NODES = {ct.NodeType.Combine: (ct.NodeCategory.MAXWELLSIM_SIMS)}
