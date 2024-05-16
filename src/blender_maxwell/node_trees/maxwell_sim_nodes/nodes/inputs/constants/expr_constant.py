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

from .... import contracts as ct
from .... import sockets
from ... import base, events


class ExprConstantNode(base.MaxwellSimNode):
	node_type = ct.NodeType.ExprConstant
	bl_label = 'Expr Constant'

	input_sockets: typ.ClassVar = {
		'Expr': sockets.ExprSocketDef(
			active_kind=ct.FlowKind.LazyValueFunc,
		),
	}
	output_sockets: typ.ClassVar = {
		'Expr': sockets.ExprSocketDef(
			active_kind=ct.FlowKind.LazyValueFunc,
			show_info_columns=True,
		),
	}

	## TODO: Allow immediately realizing any symbol, or just passing it along.
	## TODO: Alter output physical_type when the input PhysicalType changes.

	####################
	# - FlowKinds
	####################
	@events.computes_output_socket(
		# Trigger
		'Expr',
		kind=ct.FlowKind.Value,
		# Loaded
		input_sockets={'Expr'},
	)
	def compute_value(self, input_sockets: dict) -> typ.Any:
		return input_sockets['Expr']

	@events.computes_output_socket(
		# Trigger
		'Expr',
		kind=ct.FlowKind.LazyValueFunc,
		# Loaded
		input_sockets={'Expr'},
		input_socket_kinds={'Expr': ct.FlowKind.LazyValueFunc},
	)
	def compute_lazy_value_func(self, input_sockets: dict) -> typ.Any:
		return input_sockets['Expr']

	####################
	# - FlowKinds: Auxiliary
	####################
	@events.computes_output_socket(
		# Trigger
		'Expr',
		kind=ct.FlowKind.Info,
		# Loaded
		input_sockets={'Expr'},
		input_socket_kinds={'Expr': ct.FlowKind.Info},
	)
	def compute_info(self, input_sockets: dict) -> typ.Any:
		return input_sockets['Expr']

	@events.computes_output_socket(
		# Trigger
		'Expr',
		kind=ct.FlowKind.Params,
		# Loaded
		input_sockets={'Expr'},
		input_socket_kinds={'Expr': ct.FlowKind.Params},
	)
	def compute_params(self, input_sockets: dict) -> typ.Any:
		return input_sockets['Expr']


####################
# - Blender Registration
####################
BL_REGISTER = [
	ExprConstantNode,
]
BL_NODES = {ct.NodeType.ExprConstant: (ct.NodeCategory.MAXWELLSIM_INPUTS_CONSTANTS)}
