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

from .... import contracts as ct
from .... import sockets
from ... import base, events

FK = ct.FlowKind


class ExprConstantNode(base.MaxwellSimNode):
	"""An expression constant."""

	node_type = ct.NodeType.ExprConstant
	bl_label = 'Expr Constant'

	input_sockets: typ.ClassVar = {
		'Expr': sockets.ExprSocketDef(
			active_kind=FK.Func,
			show_name_selector=True,
		),
	}
	output_sockets: typ.ClassVar = {
		'Expr': sockets.ExprSocketDef(
			active_kind=FK.Func,
			show_info_columns=True,
		),
	}

	####################
	# - FlowKinds
	####################
	@events.computes_output_socket(
		# Trigger
		'Expr',
		kind=FK.Value,
		# Loaded
		inscks_kinds={'Expr': FK.Value},
	)
	def compute_value(self, input_sockets: dict) -> typ.Any:
		"""Compute the symbolic expression value."""
		return input_sockets['Expr']

	@events.computes_output_socket(
		# Trigger
		'Expr',
		kind=FK.Func,
		# Loaded
		inscks_kinds={'Expr': FK.Func},
	)
	def compute_lazy_func(self, input_sockets: dict) -> typ.Any:
		"""Compute the lazy expression function."""
		return input_sockets['Expr']

	####################
	# - FlowKinds: Auxiliary
	####################
	@events.computes_output_socket(
		# Trigger
		'Expr',
		kind=FK.Info,
		# Loaded
		input_sockets={'Expr'},
		input_socket_kinds={'Expr': FK.Info},
	)
	def compute_info(self, input_sockets: dict) -> typ.Any:
		"""Compute the tracking information flow."""
		return input_sockets['Expr']

	@events.computes_output_socket(
		# Trigger
		'Expr',
		kind=FK.Params,
		# Loaded
		input_sockets={'Expr'},
		input_socket_kinds={'Expr': FK.Params},
	)
	def compute_params(self, input_sockets: dict) -> typ.Any:
		"""Compute the fnuction parameters."""
		return input_sockets['Expr']


####################
# - Blender Registration
####################
BL_REGISTER = [
	ExprConstantNode,
]
BL_NODES = {ct.NodeType.ExprConstant: (ct.NodeCategory.MAXWELLSIM_INPUTS_CONSTANTS)}
