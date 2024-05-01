import enum
import typing as typ

import bpy
import jax.numpy as jnp
import sympy as sp

from blender_maxwell.utils import bl_cache, logger
from blender_maxwell.utils import extra_sympy_units as spux

from .... import contracts as ct
from .... import sockets
from ... import base, events

log = logger.get(__name__)

FUNCS = {
	'ADD': lambda exprs: exprs[0] + exprs[1],
	'SUB': lambda exprs: exprs[0] - exprs[1],
	'MUL': lambda exprs: exprs[0] * exprs[1],
	'DIV': lambda exprs: exprs[0] / exprs[1],
	'POW': lambda exprs: exprs[0] ** exprs[1],
	'ATAN2': lambda exprs: sp.atan2(exprs[1], exprs[0]),
	# Vector | Vector
	'VEC_VEC_DOT': lambda exprs: exprs[0].dot(exprs[1]),
	'CROSS': lambda exprs: exprs[0].cross(exprs[1]),
}

SP_FUNCS = FUNCS
JAX_FUNCS = FUNCS | {
	# Number | *
	'ATAN2': lambda exprs: jnp.atan2(exprs[1], exprs[0]),
	# Vector | Vector
	'VEC_VEC_DOT': lambda exprs: jnp.matmul(exprs[0], exprs[1]),
	'CROSS': lambda exprs: jnp.cross(exprs[0], exprs[1]),
	# Matrix | Vector
	'MAT_VEC_DOT': lambda exprs: jnp.matmul(exprs[0], exprs[1]),
	'LIN_SOLVE': lambda exprs: jnp.linalg.solve(exprs[0], exprs[1]),
	'LSQ_SOLVE': lambda exprs: jnp.linalg.lstsq(exprs[0], exprs[1]),
	# Matrix | Matrix
	'MAT_MAT_DOT': lambda exprs: jnp.matmul(exprs[0], exprs[1]),
}


class OperateMathNode(base.MaxwellSimNode):
	r"""Applies a function that depends on two inputs.

	Attributes:
		category: The category of operations to apply to the inputs.
			**Only valid** categories can be chosen.
		operation: The actual operation to apply to the inputs.
			**Only valid** operations can be chosen.
	"""

	node_type = ct.NodeType.OperateMath
	bl_label = 'Operate Math'

	input_sockets: typ.ClassVar = {
		'Expr L': sockets.ExprSocketDef(),
		'Expr R': sockets.ExprSocketDef(),
	}
	output_sockets: typ.ClassVar = {
		'Expr': sockets.ExprSocketDef(),
	}

	####################
	# - Properties
	####################
	category: enum.Enum = bl_cache.BLField(
		prop_ui=True, enum_cb=lambda self, _: self.search_categories()
	)

	operation: enum.Enum = bl_cache.BLField(
		prop_ui=True, enum_cb=lambda self, _: self.search_operations()
	)

	def search_categories(self) -> list[ct.BLEnumElement]:
		"""Deduce and return a list of valid categories for the current socket set and input data."""
		expr_l_info = self._compute_input(
			'Expr L',
			kind=ct.FlowKind.Info,
		)
		expr_r_info = self._compute_input(
			'Expr R',
			kind=ct.FlowKind.Info,
		)

		has_expr_l_info = not ct.FlowSignal.check(expr_l_info)
		has_expr_r_info = not ct.FlowSignal.check(expr_r_info)

		# Categories by Socket Set
		NUMBER_NUMBER = (
			'Number | Number',
			'Number | Number',
			'Operations between numerical elements',
		)
		NUMBER_VECTOR = (
			'Number | Vector',
			'Number | Vector',
			'Operations between numerical and vector elements',
		)
		NUMBER_MATRIX = (
			'Number | Matrix',
			'Number | Matrix',
			'Operations between numerical and matrix elements',
		)
		VECTOR_VECTOR = (
			'Vector | Vector',
			'Vector | Vector',
			'Operations between vector elements',
		)
		MATRIX_VECTOR = (
			'Matrix | Vector',
			'Matrix | Vector',
			'Operations between vector and matrix elements',
		)
		MATRIX_MATRIX = (
			'Matrix | Matrix',
			'Matrix | Matrix',
			'Operations between matrix elements',
		)
		categories = []

		if has_expr_l_info and has_expr_r_info:
			# Check Valid Broadcasting
			## Number | Number
			if expr_l_info.output_shape is None and expr_r_info.output_shape is None:
				categories = [NUMBER_NUMBER]

			## * | Number
			elif expr_r_info.output_shape is None:
				categories = []

			## Number | Vector
			elif (
				expr_l_info.output_shape is None and len(expr_r_info.output_shape) == 1
			):
				categories = [NUMBER_VECTOR]

			## Number | Matrix
			elif (
				expr_l_info.output_shape is None and len(expr_r_info.output_shape) == 2
			):
				categories = [NUMBER_MATRIX]

			## Vector | Vector
			elif (
				len(expr_l_info.output_shape) == 1
				and len(expr_r_info.output_shape) == 1
			):
				categories = [VECTOR_VECTOR]

			## Matrix | Vector
			elif (
				len(expr_l_info.output_shape) == 2  # noqa: PLR2004
				and len(expr_r_info.output_shape) == 1
			):
				categories = [MATRIX_VECTOR]

			## Matrix | Matrix
			elif (
				len(expr_l_info.output_shape) == 2  # noqa: PLR2004
				and len(expr_r_info.output_shape) == 2  # noqa: PLR2004
			):
				categories = [MATRIX_MATRIX]

		return [
			(*category, '', i) if category is not None else None
			for i, category in enumerate(categories)
		]

	def search_operations(self) -> list[ct.BLEnumElement]:
		items = []
		if self.category in ['Number | Number', 'Number | Vector', 'Number | Matrix']:
			items += [
				('ADD', 'L + R', 'Add'),
				('SUB', 'L - R', 'Subtract'),
				('MUL', 'L · R', 'Multiply'),
				('DIV', 'L / R', 'Divide'),
				('POW', 'L^R', 'Power'),
				('ATAN2', 'atan2(L,R)', 'atan2(L,R)'),
			]
		if self.category == 'Vector | Vector':
			if items:
				items += [None]
			items += [
				('VEC_VEC_DOT', 'L · R', 'Vector-Vector Product'),
				('CROSS', 'L x R', 'Cross Product'),
			]
		if self.category == 'Matrix | Vector':
			if items:
				items += [None]
			items += [
				('MAT_VEC_DOT', 'L · R', 'Matrix-Vector Product'),
				('LIN_SOLVE', 'Lx = R -> x', 'Linear Solve'),
				('LSQ_SOLVE', 'Lx = R ~> x', 'Least Squares Solve'),
			]
		if self.category == 'Matrix | Matrix':
			if items:
				items += [None]
			items += [
				('MAT_MAT_DOT', 'L · R', 'Matrix-Matrix Product'),
			]

		return [
			(*item, '', i) if item is not None else None for i, item in enumerate(items)
		]

	####################
	# - UI
	####################
	def draw_label(self):
		labels = {
			'ADD': lambda: 'L + R',
			'SUB': lambda: 'L - R',
			'MUL': lambda: 'L · R',
			'DIV': lambda: 'L / R',
			'POW': lambda: 'L^R',
			'ATAN2': lambda: 'atan2(L,R)',
		}

		if (label := labels.get(self.operation)) is not None:
			return 'Operate: ' + label()

		return self.bl_label

	def draw_props(self, _: bpy.types.Context, layout: bpy.types.UILayout) -> None:
		layout.prop(self, self.blfields['category'], text='')
		layout.prop(self, self.blfields['operation'], text='')

	####################
	# - Events
	####################
	@events.on_value_changed(
		# Trigger
		socket_name={'Expr L', 'Expr R'},
		run_on_init=True,
	)
	def on_socket_changed(self) -> None:
		# Recompute Valid Categories
		self.category = bl_cache.Signal.ResetEnumItems
		self.operation = bl_cache.Signal.ResetEnumItems

	@events.on_value_changed(
		prop_name='category',
		run_on_init=True,
	)
	def on_category_changed(self) -> None:
		self.operation = bl_cache.Signal.ResetEnumItems

	####################
	# - Output
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

		if has_expr_l_value and has_expr_r_value and operation is not None:
			return SP_FUNCS[operation]([expr_l, expr_r])

		return ct.Flowsignal.FlowPending

	@events.computes_output_socket(
		'Expr',
		kind=ct.FlowKind.LazyValueFunc,
		props={'operation'},
		input_sockets={'Expr L', 'Expr R'},
		input_socket_kinds={
			'Expr L': ct.FlowKind.LazyValueFunc,
			'Expr R': ct.FlowKind.LazyValueFunc,
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

		if has_expr_l and has_expr_r:
			return (expr_l | expr_r).compose_within(
				JAX_FUNCS[operation],
				supports_jax=True,
			)
		return ct.FlowSignal.FlowPending

	@events.computes_output_socket(
		'Expr',
		kind=ct.FlowKind.Array,
		output_sockets={'Expr'},
		output_socket_kinds={
			'Expr': {ct.FlowKind.LazyValueFunc, ct.FlowKind.Params},
		},
		unit_systems={'BlenderUnits': ct.UNITS_BLENDER},
	)
	def compute_array(self, output_sockets, unit_systems) -> ct.ArrayFlow:
		lazy_value_func = output_sockets['Expr'][ct.FlowKind.LazyValueFunc]
		params = output_sockets['Expr'][ct.FlowKind.Params]

		has_lazy_value_func = not ct.FlowSignal.check(lazy_value_func)
		has_params = not ct.FlowSignal.check(params)

		if has_lazy_value_func and has_params:
			unit_system = unit_systems['BlenderUnits']
			return ct.ArrayFlow(
				values=lazy_value_func.func_jax(
					*params.scaled_func_args(unit_system),
					**params.scaled_func_kwargs(unit_system),
				),
				unit=None,
			)

		return ct.FlowSignal.FlowPending

	####################
	# - Auxiliary: Info
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

		# Return Info of RHS
		## -> Fundamentall, this is why 'category' only has the given options.
		## -> Via 'category', we enforce that the operated-on structure is always RHS.
		## -> That makes it super duper easy to track info changes.
		if has_info_l and has_info_r and operation is not None:
			return info_r

		return ct.FlowSignal.FlowPending

	####################
	# - Auxiliary: Params
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
