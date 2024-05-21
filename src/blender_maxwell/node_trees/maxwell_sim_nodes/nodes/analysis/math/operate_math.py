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

import enum
import typing as typ

import bpy
import jax.numpy as jnp
import sympy as sp

from blender_maxwell.utils import bl_cache, logger

from .... import contracts as ct
from .... import sockets
from ... import base, events

log = logger.get(__name__)


####################
# - Operation Enum
####################
class BinaryOperation(enum.StrEnum):
	"""Valid operations for the `OperateMathNode`.

	Attributes:
		Add: Addition w/broadcasting.
		Sub: Subtraction w/broadcasting.
		Mul: Hadamard-product multiplication.
		Div: Hadamard-product based division.
		Pow: Elementwise expontiation.
		Atan2: Quadrant-respecting arctangent variant.
		VecVecDot: Dot product for vectors.
		Cross: Cross product.
		MatVecDot: Matrix-Vector dot product.
		LinSolve: Solve a linear system.
		LsqSolve: Minimize error of an underdetermined linear system.
		MatMatDot: Matrix-Matrix dot product.
	"""

	# Number | Number
	Add = enum.auto()
	Sub = enum.auto()
	Mul = enum.auto()
	Div = enum.auto()
	Pow = enum.auto()
	Atan2 = enum.auto()

	# Vector | Vector
	VecVecDot = enum.auto()
	Cross = enum.auto()

	# Matrix | Vector
	MatVecDot = enum.auto()
	LinSolve = enum.auto()
	LsqSolve = enum.auto()

	# Matrix | Matrix
	MatMatDot = enum.auto()

	####################
	# - UI
	####################
	@staticmethod
	def to_name(value: typ.Self) -> str:
		BO = BinaryOperation
		return {
			# Number | Number
			BO.Add: 'ℓ + r',
			BO.Sub: 'ℓ - r',
			BO.Mul: 'ℓ ⊙ r',  ## Notation for Hadamard Product
			BO.Div: 'ℓ / r',
			BO.Pow: 'ℓʳ',
			BO.Atan2: 'atan2(ℓ,r)',
			# Vector | Vector
			BO.VecVecDot: '𝐥 · 𝐫',
			BO.Cross: 'cross(L,R)',
			# Matrix | Vector
			BO.MatVecDot: '𝐋 · 𝐫',
			BO.LinSolve: '𝐋 ∖ 𝐫',
			BO.LsqSolve: 'argminₓ∥𝐋𝐱−𝐫∥₂',
			# Matrix | Matrix
			BO.MatMatDot: '𝐋 · 𝐑',
		}[value]

	@staticmethod
	def to_icon(value: typ.Self) -> str:
		return ''

	def bl_enum_element(self, i: int) -> ct.BLEnumElement:
		BO = BinaryOperation
		return (
			str(self),
			BO.to_name(self),
			BO.to_name(self),
			BO.to_icon(self),
			i,
		)

	####################
	# - Ops from Shape
	####################
	@staticmethod
	def by_infos(info_l: int, info_r: int) -> list[typ.Self]:
		"""Deduce valid binary operations from the shapes of the inputs."""
		BO = BinaryOperation

		ops_number_number = [
			BO.Add,
			BO.Sub,
			BO.Mul,
			BO.Div,
			BO.Pow,
			BO.Atan2,
		]

		match (info_l.output_shape_len, info_r.output_shape_len):
			# Number | *
			## Number | Number
			case (0, 0):
				return ops_number_number

			## Number | Vector
			## -> Broadcasting allows Number|Number ops to work as-is.
			case (0, 1):
				return ops_number_number

			## Number | Matrix
			## -> Broadcasting allows Number|Number ops to work as-is.
			case (0, 2):
				return ops_number_number

			# Vector | *
			## Vector | Number
			case (1, 0):
				return ops_number_number

			## Vector | Number
			case (1, 1):
				return [*ops_number_number, BO.VecVecDot, BO.Cross]

			## Vector | Matrix
			case (1, 2):
				return []

			# Matrix | *
			## Matrix | Number
			case (2, 0):
				return [*ops_number_number, BO.MatMatDot]

			## Matrix | Vector
			case (2, 1):
				return [BO.MatVecDot, BO.LinSolve, BO.LsqSolve]

			## Matrix | Matrix
			case (2, 2):
				return [*ops_number_number, BO.MatMatDot]

		return []

	####################
	# - Function Properties
	####################
	@property
	def sp_func(self):
		"""Deduce an appropriate sympy-based function that implements the binary operation for symbolic inputs."""
		BO = BinaryOperation

		## TODO: Make this compatible with sp.Matrix inputs
		return {
			# Number | Number
			BO.Add: lambda exprs: exprs[0] + exprs[1],
			BO.Sub: lambda exprs: exprs[0] - exprs[1],
			BO.Mul: lambda exprs: exprs[0] * exprs[1],
			BO.Div: lambda exprs: exprs[0] / exprs[1],
			BO.Pow: lambda exprs: exprs[0] ** exprs[1],
			BO.Atan2: lambda exprs: sp.atan2(exprs[1], exprs[0]),
		}[self]

	@property
	def jax_func(self):
		"""Deduce an appropriate jax-based function that implements the binary operation for array inputs."""
		BO = BinaryOperation

		return {
			# Number | Number
			BO.Add: lambda exprs: exprs[0] + exprs[1],
			BO.Sub: lambda exprs: exprs[0] - exprs[1],
			BO.Mul: lambda exprs: exprs[0] * exprs[1],
			BO.Div: lambda exprs: exprs[0] / exprs[1],
			BO.Pow: lambda exprs: exprs[0] ** exprs[1],
			BO.Atan2: lambda exprs: sp.atan2(exprs[1], exprs[0]),
			# Vector | Vector
			BO.VecVecDot: lambda exprs: jnp.dot(exprs[0], exprs[1]),
			BO.Cross: lambda exprs: jnp.cross(exprs[0], exprs[1]),
			# Matrix | Vector
			BO.MatVecDot: lambda exprs: jnp.matmul(exprs[0], exprs[1]),
			BO.LinSolve: lambda exprs: jnp.linalg.solve(exprs[0], exprs[1]),
			BO.LsqSolve: lambda exprs: jnp.linalg.lstsq(exprs[0], exprs[1]),
			# Matrix | Matrix
			BO.MatMatDot: lambda exprs: jnp.matmul(exprs[0], exprs[1]),
		}[self]

	####################
	# - InfoFlow Transform
	####################
	def transform_infos(self, info_l: ct.InfoFlow, info_r: ct.InfoFlow):
		BO = BinaryOperation

		info_largest = (
			info_l if info_l.output_shape_len > info_l.output_shape_len else info_l
		)
		info_any = info_largest
		return {
			# Number | * or * | Number
			BO.Add: info_largest,
			BO.Sub: info_largest,
			BO.Mul: info_largest,
			BO.Div: info_largest,
			BO.Pow: info_largest,
			BO.Atan2: info_largest,
			# Vector | Vector
			BO.VecVecDot: info_any,
			BO.Cross: info_any,
			# Matrix | Vector
			BO.MatVecDot: info_r,
			BO.LinSolve: info_r,
			BO.LsqSolve: info_r,
			# Matrix | Matrix
			BO.MatMatDot: info_any,
		}[self]


####################
# - Node
####################
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
	# - Properties
	####################
	@events.on_value_changed(
		socket_name={'Expr L', 'Expr R'},
		input_sockets={'Expr L', 'Expr R'},
		input_socket_kinds={'Expr L': ct.FlowKind.Info, 'Expr R': ct.FlowKind.Info},
		input_sockets_optional={'Expr L': True, 'Expr R': True},
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
		info_l = self._compute_input('Expr L', kind=ct.FlowKind.Info)
		info_r = self._compute_input('Expr R', kind=ct.FlowKind.Info)

		has_info_l = not ct.FlowSignal.check(info_l)
		has_info_r = not ct.FlowSignal.check(info_r)

		if has_info_l and has_info_r:
			return (info_l, info_r)

		return None

	operation: BinaryOperation = bl_cache.BLField(
		enum_cb=lambda self, _: self.search_operations(),
		cb_depends_on={'expr_infos'},
	)

	def search_operations(self) -> list[ct.BLEnumElement]:
		if self.expr_infos is not None:
			return [
				operation.bl_enum_element(i)
				for i, operation in enumerate(
					BinaryOperation.by_infos(*self.expr_infos)
				)
			]
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
			return 'Op: ' + BinaryOperation.to_name(self.operation)

		return self.bl_label

	def draw_props(self, _: bpy.types.Context, layout: bpy.types.UILayout) -> None:
		"""Draw node properties in the node.

		Parameters:
			col: UI target for drawing.
		"""
		layout.prop(self, self.blfields['operation'], text='')

	####################
	# - FlowKind.Value|Func
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
		operation = props['operation']
		expr_l = input_sockets['Expr L']
		expr_r = input_sockets['Expr R']

		has_expr_l_value = not ct.FlowSignal.check(expr_l)
		has_expr_r_value = not ct.FlowSignal.check(expr_r)

		# Compute Sympy Function
		## -> The operation enum directly provides the appropriate function.
		if has_expr_l_value and has_expr_r_value and operation is not None:
			operation.sp_func([expr_l, expr_r])

		return ct.Flowsignal.FlowPending

	@events.computes_output_socket(
		'Expr',
		kind=ct.FlowKind.Func,
		props={'operation'},
		input_sockets={'Expr L', 'Expr R'},
		input_socket_kinds={
			'Expr L': ct.FlowKind.Func,
			'Expr R': ct.FlowKind.Func,
		},
	)
	def compose_func(self, props: dict, input_sockets: dict):
		operation = props['operation']
		if operation is None:
			return ct.FlowSignal.FlowPending

		expr_l = input_sockets['Expr L']
		expr_r = input_sockets['Expr R']

		has_expr_l = not ct.FlowSignal.check(expr_l)
		has_expr_r = not ct.FlowSignal.check(expr_r)

		# Compute Jax Function
		## -> The operation enum directly provides the appropriate function.
		if has_expr_l and has_expr_r:
			return (expr_l | expr_r).compose_within(
				operation.jax_func,
				supports_jax=True,
			)
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
		operation = props['operation']
		info_l = input_sockets['Expr L']
		info_r = input_sockets['Expr R']

		has_info_l = not ct.FlowSignal.check(info_l)
		has_info_r = not ct.FlowSignal.check(info_r)

		# Compute Info
		## -> The operation enum directly provides the appropriate transform.
		if (
			has_info_l
			and has_info_r
			and operation is not None
			and operation in BinaryOperation.by_infos(info_l, info_r)
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
		operation = props['operation']
		params_l = input_sockets['Expr L']
		params_r = input_sockets['Expr R']

		has_params_l = not ct.FlowSignal.check(params_l)
		has_params_r = not ct.FlowSignal.check(params_r)

		# Compute Params
		## -> Operations don't add new parameters, so just concatenate L|R.
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
