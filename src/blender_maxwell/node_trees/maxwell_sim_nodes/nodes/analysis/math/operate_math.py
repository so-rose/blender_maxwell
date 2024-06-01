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

"""Implements the `OperateMathNode`.

See `blender_maxwell.maxwell_sim_nodes.math_system` for the actual mathematics implementation.
"""

import typing as typ

import bpy

from blender_maxwell.utils import bl_cache, logger

from .... import contracts as ct
from .... import math_system, sockets
from ... import base, events

log = logger.get(__name__)


class OperateMathNode(base.MaxwellSimNode):
	r"""Applies a binary function between two expressions.

	Attributes:
		category: The category of operations to apply to the inputs.
			**Only valid** categories can be chosen.
		operation: The actual operation to apply to the inputs.
			**Only valid** operations can be chosen.
	"""

	node_type = ct.NodeType.OperateMath
	bl_label = 'Operate Math'

	input_sockets: typ.ClassVar = {
		'Expr L': sockets.ExprSocketDef(active_kind=ct.FlowKind.Func),
		'Expr R': sockets.ExprSocketDef(active_kind=ct.FlowKind.Func),
	}
	output_sockets: typ.ClassVar = {
		'Expr': sockets.ExprSocketDef(
			active_kind=ct.FlowKind.Func, show_info_columns=True
		),
	}

	####################
	# - Properties: Incoming InfoFlows
	####################
	@events.on_value_changed(
		# Trigger
		socket_name={'Expr L', 'Expr R'},
		# Loaded
		input_sockets={'Expr L', 'Expr R'},
		input_socket_kinds={'Expr L': ct.FlowKind.Info, 'Expr R': ct.FlowKind.Info},
		input_sockets_optional={'Expr L': True, 'Expr R': True},
		# Flow
		## -> See docs in TransformMathNode
		stop_propagation=True,
	)
	def on_input_exprs_changed(self, input_sockets) -> None:  # noqa: D102
		has_info_l = not ct.FlowSignal.check(input_sockets['Expr L'])
		has_info_r = not ct.FlowSignal.check(input_sockets['Expr R'])

		info_l_pending = ct.FlowSignal.check_single(
			input_sockets['Expr L'], ct.FlowSignal.FlowPending
		)
		info_r_pending = ct.FlowSignal.check_single(
			input_sockets['Expr R'], ct.FlowSignal.FlowPending
		)

		if has_info_l and has_info_r and not info_l_pending and not info_r_pending:
			self.expr_infos = bl_cache.Signal.InvalidateCache

	@bl_cache.cached_bl_property()
	def expr_infos(self) -> tuple[ct.InfoFlow, ct.InfoFlow] | None:
		"""Computed `InfoFlow`s of both expressions."""
		info_l = self._compute_input('Expr L', kind=ct.FlowKind.Info)
		info_r = self._compute_input('Expr R', kind=ct.FlowKind.Info)

		has_info_l = not ct.FlowSignal.check(info_l)
		has_info_r = not ct.FlowSignal.check(info_r)

		if has_info_l and has_info_r:
			return (info_l, info_r)

		return None

	####################
	# - Property: Operation
	####################
	operation: math_system.BinaryOperation = bl_cache.BLField(
		enum_cb=lambda self, _: self.search_operations(),
		cb_depends_on={'expr_infos'},
	)

	def search_operations(self) -> list[ct.BLEnumElement]:
		"""Retrieve valid operations based on the input `InfoFlow`s."""
		if self.expr_infos is not None:
			return math_system.BinaryOperation.bl_enum_elements(*self.expr_infos)
		return []

	####################
	# - UI
	####################
	def draw_label(self):
		"""Show the current operation (if any) in the node's header label.

		Notes:
			Called by Blender to determine the text to place in the node's header.
		"""
		if self.operation is not None:
			return 'Op: ' + math_system.BinaryOperation.to_name(self.operation)

		return self.bl_label

	def draw_props(self, _: bpy.types.Context, layout: bpy.types.UILayout) -> None:
		"""Draw node properties in the node.

		Parameters:
			col: UI target for drawing.
		"""
		layout.prop(self, self.blfields['operation'], text='')

	####################
	# - FlowKind.Value
	####################
	@events.computes_output_socket(
		'Expr',
		kind=ct.FlowKind.Value,
		props={'operation'},
		input_sockets={'Expr L', 'Expr R'},
		input_socket_kinds={
			'Expr L': ct.FlowKind.Value,
			'Expr R': ct.FlowKind.Value,
		},
	)
	def compute_value(self, props: dict, input_sockets: dict):
		"""Binary operation on two symbolic input expressions."""
		expr_l = input_sockets['Expr L']
		expr_r = input_sockets['Expr R']

		has_expr_l_value = not ct.FlowSignal.check(expr_l)
		has_expr_r_value = not ct.FlowSignal.check(expr_r)

		operation = props['operation']
		if has_expr_l_value and has_expr_r_value and operation is not None:
			return operation.sp_func([expr_l, expr_r])

		return ct.FlowSignal.FlowPending

	####################
	# - FlowKind.Func
	####################
	@events.computes_output_socket(
		'Expr',
		kind=ct.FlowKind.Func,
		# Loaded
		props={'operation'},
		input_sockets={'Expr L', 'Expr R'},
		input_socket_kinds={
			'Expr L': ct.FlowKind.Func,
			'Expr R': ct.FlowKind.Func,
		},
		output_sockets={'Expr'},
		output_socket_kinds={'Expr': ct.FlowKind.Info},
	)
	def compute_func(self, props, input_sockets, output_sockets):
		"""Binary operation on two lazy-defined input expressions."""
		expr_l = input_sockets['Expr L']
		expr_r = input_sockets['Expr R']
		output_info = output_sockets['Expr']

		has_expr_l = not ct.FlowSignal.check(expr_l)
		has_expr_r = not ct.FlowSignal.check(expr_r)
		has_output_info = not ct.FlowSignal.check(output_info)

		operation = props['operation']
		if operation is not None and has_expr_l and has_expr_r and has_output_info:
			return self.operation.transform_funcs(expr_l, expr_r)
		return ct.FlowSignal.FlowPending

	####################
	# - FlowKind.Info
	####################
	@events.computes_output_socket(
		'Expr',
		kind=ct.FlowKind.Info,
		props={'operation'},
		input_sockets={'Expr L', 'Expr R'},
		input_socket_kinds={
			'Expr L': ct.FlowKind.Info,
			'Expr R': ct.FlowKind.Info,
		},
	)
	def compute_info(self, props, input_sockets) -> ct.InfoFlow:
		"""Transform the input information of both lazy inputs."""
		info_l = input_sockets['Expr L']
		info_r = input_sockets['Expr R']

		has_info_l = not ct.FlowSignal.check(info_l)
		has_info_r = not ct.FlowSignal.check(info_r)

		operation = props['operation']
		if (
			has_info_l and has_info_r and operation is not None
			# and operation in BO.by_infos(info_l, info_r)
		):
			return operation.transform_infos(info_l, info_r)

		return ct.FlowSignal.FlowPending

	####################
	# - FlowKind.Params
	####################
	@events.computes_output_socket(
		'Expr',
		kind=ct.FlowKind.Params,
		props={'operation'},
		input_sockets={'Expr L', 'Expr R'},
		input_socket_kinds={
			'Expr L': ct.FlowKind.Params,
			'Expr R': ct.FlowKind.Params,
		},
	)
	def compute_params(self, props, input_sockets) -> ct.ParamsFlow | ct.FlowSignal:
		"""Merge the lazy input parameters."""
		params_l = input_sockets['Expr L']
		params_r = input_sockets['Expr R']

		has_params_l = not ct.FlowSignal.check(params_l)
		has_params_r = not ct.FlowSignal.check(params_r)

		operation = props['operation']
		if has_params_l and has_params_r and operation is not None:
			return params_l | params_r

		return ct.FlowSignal.FlowPending


####################
# - Blender Registration
####################
BL_REGISTER = [
	OperateMathNode,
]
BL_NODES = {ct.NodeType.OperateMath: (ct.NodeCategory.MAXWELLSIM_ANALYSIS_MATH)}
