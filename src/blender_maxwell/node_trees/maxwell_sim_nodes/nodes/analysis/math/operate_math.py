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

	input_socket_sets: typ.ClassVar = {
		'Expr | Expr': {
			'Expr L': sockets.ExprSocketDef(),
			'Expr R': sockets.ExprSocketDef(),
		},
		'Data | Data': {
			'Data L': sockets.DataSocketDef(
				format='jax', default_show_info_columns=False
			),
			'Data R': sockets.DataSocketDef(
				format='jax', default_show_info_columns=False
			),
		},
		'Expr | Data': {
			'Expr L': sockets.ExprSocketDef(),
			'Data R': sockets.DataSocketDef(
				format='jax', default_show_info_columns=False
			),
		},
	}
	output_socket_sets: typ.ClassVar = {
		'Expr | Expr': {
			'Expr': sockets.ExprSocketDef(),
		},
		'Data | Data': {
			'Data': sockets.DataSocketDef(
				format='jax', default_show_info_columns=False
			),
		},
		'Expr | Data': {
			'Data': sockets.DataSocketDef(
				format='jax', default_show_info_columns=False
			),
		},
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
		data_l_info = self._compute_input(
			'Data L', kind=ct.FlowKind.Info, optional=True
		)
		data_r_info = self._compute_input(
			'Data R', kind=ct.FlowKind.Info, optional=True
		)

		has_data_l_info = not ct.FlowSignal.check(data_l_info)
		has_data_r_info = not ct.FlowSignal.check(data_r_info)

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

		## Expr | Expr
		if self.active_socket_set == 'Expr | Expr':
			return [NUMBER_NUMBER]

		## Data | Data
		if (
			self.active_socket_set == 'Data | Data'
			and has_data_l_info
			and has_data_r_info
		):
			# Check Valid Broadcasting
			## Number | Number
			if data_l_info.output_shape is None and data_r_info.output_shape is None:
				categories = [NUMBER_NUMBER]

			## Number | Vector
			elif (
				data_l_info.output_shape is None and len(data_r_info.output_shape) == 1
			):
				categories = [NUMBER_VECTOR]

			## Number | Matrix
			elif (
				data_l_info.output_shape is None and len(data_r_info.output_shape) == 2
			):  # noqa: PLR2004
				categories = [NUMBER_MATRIX]

			## Vector | Vector
			elif (
				len(data_l_info.output_shape) == 1
				and len(data_r_info.output_shape) == 1
			):
				categories = [VECTOR_VECTOR]

			## Matrix | Vector
			elif (
				len(data_l_info.output_shape) == 2  # noqa: PLR2004
				and len(data_r_info.output_shape) == 1
			):
				categories = [MATRIX_VECTOR]

			## Matrix | Matrix
			elif (
				len(data_l_info.output_shape) == 2  # noqa: PLR2004
				and len(data_r_info.output_shape) == 2  # noqa: PLR2004
			):
				categories = [MATRIX_MATRIX]

		## Expr | Data
		if self.active_socket_set == 'Expr | Data' and has_data_r_info:
			if data_r_info.output_shape is None:
				categories = [NUMBER_NUMBER]
			else:
				categories = {
					1: [NUMBER_NUMBER, NUMBER_VECTOR],
					2: [NUMBER_NUMBER, NUMBER_MATRIX],
				}[len(data_r_info.output_shape)]

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
				('DIV', 'L ÷ R', 'Divide'),
				('POW', 'L^R', 'Power'),
				('ATAN2', 'atan2(L,R)', 'atan2(L,R)'),
			]
		if self.category in 'Vector | Vector':
			if items:
				items += [None]
			items += [
				('VEC_VEC_DOT', 'L · R', 'Vector-Vector Product'),
				('CROSS', 'L x R', 'Cross Product'),
				('PROJ', 'proj(L, R)', 'Projection'),
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

	def draw_props(self, _: bpy.types.Context, layout: bpy.types.UILayout) -> None:
		layout.prop(self, self.blfields['category'], text='')
		layout.prop(self, self.blfields['operation'], text='')

	####################
	# - Events
	####################
	@events.on_value_changed(
		# Trigger
		socket_name={'Expr L', 'Expr R', 'Data L', 'Data R'},
		prop_name='active_socket_set',
		run_on_init=True,
	)
	def on_socket_set_changed(self) -> None:
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
	)
	def compute_expr(self, props: dict, input_sockets: dict):
		expr_l = input_sockets['Expr L']
		expr_r = input_sockets['Expr R']

		return {
			'ADD': lambda: expr_l + expr_r,
			'SUB': lambda: expr_l - expr_r,
			'MUL': lambda: expr_l * expr_r,
			'DIV': lambda: expr_l / expr_r,
			'POW': lambda: expr_l**expr_r,
			'ATAN2': lambda: sp.atan2(expr_r, expr_l),
		}[props['operation']]()

	@events.computes_output_socket(
		'Data',
		kind=ct.FlowKind.LazyValueFunc,
		props={'operation'},
		input_sockets={'Data L', 'Data R'},
		input_socket_kinds={
			'Data L': ct.FlowKind.LazyValueFunc,
			'Data R': ct.FlowKind.LazyValueFunc,
		},
		input_sockets_optional={
			'Data L': True,
			'Data R': True,
		},
	)
	def compute_data(self, props: dict, input_sockets: dict):
		data_l = input_sockets['Data L']
		data_r = input_sockets['Data R']
		has_data_l = not ct.FlowSignal.check(data_l)

		mapping_func = {
			# Number | *
			'ADD': lambda datas: datas[0] + datas[1],
			'SUB': lambda datas: datas[0] - datas[1],
			'MUL': lambda datas: datas[0] * datas[1],
			'DIV': lambda datas: datas[0] / datas[1],
			'POW': lambda datas: datas[0] ** datas[1],
			'ATAN2': lambda datas: jnp.atan2(datas[1], datas[0]),
			# Vector | Vector
			'VEC_VEC_DOT': lambda datas: jnp.matmul(datas[0], datas[1]),
			'CROSS': lambda datas: jnp.cross(datas[0], datas[1]),
			# Matrix | Vector
			'MAT_VEC_DOT': lambda datas: jnp.matmul(datas[0], datas[1]),
			'LIN_SOLVE': lambda datas: jnp.linalg.solve(datas[0], datas[1]),
			'LSQ_SOLVE': lambda datas: jnp.linalg.lstsq(datas[0], datas[1]),
			# Matrix | Matrix
			'MAT_MAT_DOT': lambda datas: jnp.matmul(datas[0], datas[1]),
		}[props['operation']]

		# Compose by Socket Set
		## Data | Data
		if has_data_l:
			return (data_l | data_r).compose_within(
				mapping_func,
				supports_jax=True,
			)

		## Expr | Data
		expr_l_lazy_value_func = ct.LazyValueFuncFlow(
			func=lambda expr_l_value: expr_l_value,
			func_args=[typ.Any],
			supports_jax=True,
		)
		return (expr_l_lazy_value_func | data_r).compose_within(
			mapping_func,
			supports_jax=True,
		)

	@events.computes_output_socket(
		'Data',
		kind=ct.FlowKind.Array,
		output_sockets={'Data'},
		output_socket_kinds={
			'Data': {ct.FlowKind.LazyValueFunc, ct.FlowKind.Params},
		},
	)
	def compute_array(self, output_sockets: dict) -> ct.ArrayFlow:
		lazy_value_func = output_sockets['Data'][ct.FlowKind.LazyValueFunc]
		params = output_sockets['Data'][ct.FlowKind.Params]

		has_lazy_value_func = not ct.FlowSignal.check(lazy_value_func)
		has_params = not ct.FlowSignal.check(params)

		if has_lazy_value_func and has_params:
			return ct.ArrayFlow(
				values=lazy_value_func.func_jax(
					*params.func_args, **params.func_kwargs
				),
				unit=None,
			)

		return ct.FlowSignal.FlowPending

	####################
	# - Auxiliary: Params
	####################
	@events.computes_output_socket(
		'Data',
		kind=ct.FlowKind.Params,
		props={'operation'},
		input_sockets={'Expr L', 'Data L', 'Data R'},
		input_socket_kinds={
			'Expr L': ct.FlowKind.Value,
			'Data L': {ct.FlowKind.Info, ct.FlowKind.Params},
			'Data R': {ct.FlowKind.Info, ct.FlowKind.Params},
		},
		input_sockets_optional={
			'Expr L': True,
			'Data L': True,
			'Data R': True,
		},
	)
	def compute_data_params(
		self, props, input_sockets
	) -> ct.ParamsFlow | ct.FlowSignal:
		expr_l = input_sockets['Expr L']
		data_l_info = input_sockets['Data L'][ct.FlowKind.Info]
		data_l_params = input_sockets['Data L'][ct.FlowKind.Params]
		data_r_info = input_sockets['Data R'][ct.FlowKind.Info]
		data_r_params = input_sockets['Data R'][ct.FlowKind.Params]

		has_expr_l = not ct.FlowSignal.check(expr_l)
		has_data_l_info = not ct.FlowSignal.check(data_l_info)
		has_data_l_params = not ct.FlowSignal.check(data_l_params)
		has_data_r_info = not ct.FlowSignal.check(data_r_info)
		has_data_r_params = not ct.FlowSignal.check(data_r_params)

		#log.critical((props, input_sockets))

		# Compose by Socket Set
		## Data | Data
		if (
			has_data_l_info
			and has_data_l_params
			and has_data_r_info
			and has_data_r_params
		):
			return data_l_params | data_r_params

		## Expr | Data
		if has_expr_l and has_data_r_info and has_data_r_params:
			operation = props['operation']
			data_unit = data_r_info.output_unit

			# By Operation
			## Add/Sub: Scale to Output Unit
			if operation in ['ADD', 'SUB', 'MUL', 'DIV']:
				if not spux.uses_units(expr_l):
					value = spux.sympy_to_python(expr_l)
				else:
					value = spux.sympy_to_python(spux.scale_to_unit(expr_l, data_unit))

				return data_r_params.compose_within(
					enclosing_func_args=[value],
				)

			## Pow: Doesn't Exist (?)
			## -> See https://math.stackexchange.com/questions/4326081/units-of-the-exponential-function
			if operation == 'POW':
				return ct.FlowSignal.FlowPending

			## atan2(): Only Length
			## -> Implicitly presume that Data L/R use length units.
			if operation == 'ATAN2':
				if not spux.uses_units(expr_l):
					value = spux.sympy_to_python(expr_l)
				else:
					value = spux.sympy_to_python(spux.scale_to_unit(expr_l, data_unit))

				return data_r_params.compose_within(
					enclosing_func_args=[value],
				)

			return data_r_params.compose_within(
				enclosing_func_args=[
					spux.sympy_to_python(spux.scale_to_unit(expr_l, data_unit))
				]
			)

		return ct.FlowSignal.FlowPending

	####################
	# - Auxiliary: Info
	####################
	@events.computes_output_socket(
		'Data',
		kind=ct.FlowKind.Info,
		input_sockets={'Expr L', 'Data L', 'Data R'},
		input_socket_kinds={
			'Expr L': ct.FlowKind.Value,
			'Data L': ct.FlowKind.Info,
			'Data R': ct.FlowKind.Info,
		},
		input_sockets_optional={
			'Expr L': True,
			'Data L': True,
			'Data R': True,
		},
	)
	def compute_data_info(self, input_sockets: dict) -> ct.InfoFlow:
		expr_l = input_sockets['Expr L']
		data_l_info = input_sockets['Data L']
		data_r_info = input_sockets['Data R']

		has_expr_l = not ct.FlowSignal.check(expr_l)
		has_data_l_info = not ct.FlowSignal.check(data_l_info)
		has_data_r_info = not ct.FlowSignal.check(data_r_info)

		# Info by Socket Set
		## Data | Data
		if has_data_l_info and has_data_r_info:
			return data_r_info

		## Expr | Data
		if has_expr_l and has_data_r_info:
			return data_r_info

		return ct.FlowSignal.FlowPending


####################
# - Blender Registration
####################
BL_REGISTER = [
	OperateMathNode,
]
BL_NODES = {ct.NodeType.OperateMath: (ct.NodeCategory.MAXWELLSIM_ANALYSIS_MATH)}
