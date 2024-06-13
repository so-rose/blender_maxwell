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
from blender_maxwell.utils import sympy_extra as spux

from .... import contracts as ct
from .... import math_system, sockets
from ... import base, events

log = logger.get(__name__)

FK = ct.FlowKind
FS = ct.FlowSignal
BO = math_system.BinaryOperation
MT = spux.MathType


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
		'Expr L': sockets.ExprSocketDef(active_kind=FK.Func),
		'Expr R': sockets.ExprSocketDef(active_kind=FK.Func),
	}
	output_sockets: typ.ClassVar = {
		'Expr': sockets.ExprSocketDef(active_kind=FK.Func, show_info_columns=True),
	}

	####################
	# - Properties: Incoming InfoFlows
	####################
	@events.on_value_changed(
		# Trigger
		socket_name={'Expr L': FK.Info, 'Expr R': FK.Info},
		# Loaded
		inscks_kinds={'Expr L': FK.Info, 'Expr R': FK.Info},
		input_sockets_optional={'Expr L', 'Expr R'},
		# Flow
		## -> See docs in TransformMathNode
		stop_propagation=True,
	)
	def on_input_exprs_changed(self, input_sockets) -> None:
		"""Queue an update of the cached expression infos whenever data changed."""
		has_info_l = not FS.check(input_sockets['Expr L'])
		has_info_r = not FS.check(input_sockets['Expr R'])

		info_l_pending = FS.check_single(input_sockets['Expr L'], FS.FlowPending)
		info_r_pending = FS.check_single(input_sockets['Expr R'], FS.FlowPending)

		if has_info_l and has_info_r and not info_l_pending and not info_r_pending:
			self.expr_infos = bl_cache.Signal.InvalidateCache

	@bl_cache.cached_bl_property()
	def expr_infos(self) -> tuple[ct.InfoFlow, ct.InfoFlow] | None:
		"""Computed `InfoFlow`s of both expressions."""
		info_l = self._compute_input('Expr L', kind=FK.Info)
		info_r = self._compute_input('Expr R', kind=FK.Info)

		has_info_l = not FS.check(info_l)
		has_info_r = not FS.check(info_r)

		if has_info_l and has_info_r:
			return (info_l, info_r)
		return None

	####################
	# - Property: Operation
	####################
	operation: BO = bl_cache.BLField(
		enum_cb=lambda self, _: self.search_operations(),
		cb_depends_on={'expr_infos'},
	)

	def search_operations(self) -> list[ct.BLEnumElement]:
		"""Retrieve valid operations based on the input `InfoFlow`s."""
		if self.expr_infos is not None:
			return BO.bl_enum_elements(*self.expr_infos)
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
			return 'Op: ' + BO.to_name(self.operation)

		return self.bl_label

	def draw_props(self, _: bpy.types.Context, layout: bpy.types.UILayout) -> None:
		"""Draw properties in the node.

		Parameters:
			col: UI target for drawing.
		"""
		layout.prop(self, self.blfields['operation'], text='')

	####################
	# - FlowKind.Value
	####################
	@events.computes_output_socket(
		'Expr',
		kind=FK.Value,
		# Loaded
		props={'operation'},
		inscks_kinds={
			'Expr L': FK.Value,
			'Expr R': FK.Value,
		},
	)
	def compute_value(self, props, input_sockets) -> ct.InfoFlow | FS:
		"""Binary operation on two symbolic input expressions."""
		expr_l = input_sockets['Expr L']
		expr_r = input_sockets['Expr R']

		operation = props['operation']
		if operation is not None:
			return operation.sp_func([expr_l, expr_r])
		return FS.FlowPending

	####################
	# - FlowKind.Func
	####################
	@events.computes_output_socket(
		'Expr',
		kind=FK.Func,
		# Loaded
		props={'operation'},
		inscks_kinds={
			'Expr L': FK.Func,
			'Expr R': FK.Func,
		},
	)
	def compute_func(self, props, input_sockets) -> ct.InfoFlow | FS:
		"""Binary operation on two lazy-defined input expressions."""
		expr_l = input_sockets['Expr L']
		expr_r = input_sockets['Expr R']

		operation = props['operation']
		if operation is not None:
			return self.operation.transform_funcs(expr_l, expr_r)
		return FS.FlowPending

	####################
	# - FlowKind.Info
	####################
	@events.computes_output_socket(
		'Expr',
		kind=FK.Info,
		# Loaded
		props={'operation'},
		input_sockets={'Expr L', 'Expr R'},
		input_socket_kinds={
			'Expr L': FK.Info,
			'Expr R': FK.Info,
		},
	)
	def compute_info(self, props, input_sockets) -> ct.InfoFlow:
		"""Transform the input information of both lazy inputs."""
		info_l = input_sockets['Expr L']
		info_r = input_sockets['Expr R']

		has_info_l = not FS.check(info_l)
		has_info_r = not FS.check(info_r)

		operation = props['operation']
		if (
			has_info_l
			and has_info_r
			and operation is not None
			and operation in BO.from_infos(info_l, info_r)
		):
			return operation.transform_infos(info_l, info_r)

		return FS.FlowPending

	####################
	# - FlowKind.Params
	####################
	@events.computes_output_socket(
		'Expr',
		kind=FK.Params,
		# Loaded
		props={'operation'},
		input_sockets={'Expr L', 'Expr R'},
		input_socket_kinds={
			'Expr L': FK.Params,
			'Expr R': FK.Params,
		},
	)
	def compute_params(self, props, input_sockets) -> ct.ParamsFlow | FS:
		"""Merge the lazy input parameters."""
		params_l = input_sockets['Expr L']
		params_r = input_sockets['Expr R']

		operation = props['operation']
		if operation is not None:
			return operation.transform_params(params_l, params_r)
		return FS.FlowPending


####################
# - Blender Registration
####################
BL_REGISTER = [
	OperateMathNode,
]
BL_NODES = {ct.NodeType.OperateMath: (ct.NodeCategory.MAXWELLSIM_ANALYSIS_MATH)}
