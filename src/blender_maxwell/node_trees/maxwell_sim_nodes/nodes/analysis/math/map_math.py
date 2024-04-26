"""Declares `MapMathNode`."""

import enum
import typing as typ

import bpy
import jax
import jax.numpy as jnp
import sympy as sp

from blender_maxwell.utils import bl_cache, logger
from blender_maxwell.utils import extra_sympy_units as spux

from .... import contracts as ct
from .... import sockets
from ... import base, events

log = logger.get(__name__)

X_COMPLEX = sp.Symbol('x', complex=True)


class MapMathNode(base.MaxwellSimNode):
	r"""Applies a function by-structure to the data.

	The shape, type, and interpretation of the input/output data is dynamically shown.

	# Socket Sets
	The line between a "map" and a "filter" is generally a matter of taste.
	In general, "map" provides "do something to each of x" operations.

	While it is often generally assumed that `numpy` broadcasting rules are well-known, dimensional data is inherently complicated.
	Therefore, we choose an explicit, opinionated approach to "how things are mapped", prioritizing predictability over flexibility.

	## By Element
	Applies a function to each scalar number of the array.

	:::{.callout-tip title="Example"}
	Say we have a standard `(50, 3)` array with a `float32` (`f32`) datatype.
	We could interpret such an indexed structure as an **element map**:

	$$
		A:\,\,\underbrace{(\mathbb{Z}_{50}, \mathbb{Z}_3)}_{\texttt{(50,3)}} \to \underbrace{(\mathbb{R})}_{\texttt{f32}}
	$$

	`By Element` simply applies a function to each output value, $\mathbb{R}$, producing a new $A$ with the same dimensions.
	Note that the datatype might be altered, ex. `\mathbb{C} \to \mathbb{R}`, as part of the function.
	:::


	## By Vector
	Applies a function to each vector, the elements of which span the **last axis**.

	This **might** produce a well-known dimensionality change, depending on what each vector maps to.

	:::{.callout-tip title="Example"}
	Let's build on the `By Element` example, by interpreting it as a list of column vectors, and taking the length of each.

	`By Vector` operates on the same data, but interpreted in a slightly deconstructed way:

	$$
		A:\,\,\underbrace{(\mathbb{Z}_{50})}_{\texttt{(50,)}} \to (\underbrace{(\mathbb{Z}_3)}_{\texttt{(3,)}} \to \underbrace{(\mathbb{R})}_{\texttt{f32}})
	$$

	`By Vector` applies a function to each $\underbrace{(\mathbb{Z}_3)}_{\texttt{(3,)}} \to \underbrace{(\mathbb{R})}_{\texttt{f32}}$.
	Applying a standard 2-norm

	$$
		||\cdot||_2:\,\,\,\,(\underbrace{(\mathbb{Z}_3)}_{\texttt{(3,)}} \to \underbrace{(\mathbb{R})}_{\texttt{f32}}) \to \underbrace{(\mathbb{R})}_{\texttt{f32}}
	$$

	to our $A$ results in a new, reduced-dimension array:

	$$
		A_{||\cdot||_2}:\,\,\underbrace{(\mathbb{Z}_{50})}_{\texttt{(50,)}} \to \underbrace{(\mathbb{R})}_{\texttt{f32}}
	$$
	:::


	## By Matrix
	Applies a function to each matrix, the elements of which span the **last two axes**.

	This **might** produce a well-known dimensionality change, depending on what each matrix maps to.

	:::{.callout-tip title="Just Like Vectors"}
	At this point, we reach 3D, and mental models become more difficult.

	When dealing with high-dimensional arrays, it is suggested to draw out the math, ex. with the explicit notation introduced earlier.
	:::

	## Expr
	Applies a user-sourced symbolic expression to a single symbol, with the symbol either representing (selectably) a single element, vector, or matrix.
	The name and type of the available symbol is clearly shown, and most valid `sympy` expressions that you would expect to work, should work.

	Use of expressions generally imposes no performance penalty: Just like the baked-in operations, it is compiled to a high-performance `jax` function.
	Thus, it participates in the `ct.FlowKind.LazyValueFunc` composition chain.


	Attributes:
		operation: Operation to apply to the input.
	"""

	node_type = ct.NodeType.MapMath
	bl_label = 'Map Math'

	input_sockets: typ.ClassVar = {
		'Data': sockets.DataSocketDef(format='jax'),
	}
	input_socket_sets: typ.ClassVar = {
		'By Element': {},
		'By Vector': {},
		'By Matrix': {},
		'Expr': {
			'Mapper': sockets.ExprSocketDef(
				complex_symbols=[X_COMPLEX],
				default_expr=X_COMPLEX,
			),
		},
	}
	output_sockets: typ.ClassVar = {
		'Data': sockets.DataSocketDef(format='jax'),
	}

	####################
	# - Properties
	####################
	operation: enum.Enum = bl_cache.BLField(
		prop_ui=True, enum_cb=lambda self, _: self.search_operations()
	)

	def search_operations(self) -> list[ct.BLEnumElement]:
		if self.active_socket_set == 'By Element':
			items = [
				# General
				('REAL', 'ℝ(v)', 'real(v) (by el)'),
				('IMAG', 'Im(v)', 'imag(v) (by el)'),
				('ABS', '|v|', 'abs(v) (by el)'),
				('SQ', 'v²', 'v^2 (by el)'),
				('SQRT', '√v', 'sqrt(v) (by el)'),
				('INV_SQRT', '1/√v', '1/sqrt(v) (by el)'),
				None,
				# Trigonometry
				('COS', 'cos v', 'cos(v) (by el)'),
				('SIN', 'sin v', 'sin(v) (by el)'),
				('TAN', 'tan v', 'tan(v) (by el)'),
				('ACOS', 'acos v', 'acos(v) (by el)'),
				('ASIN', 'asin v', 'asin(v) (by el)'),
				('ATAN', 'atan v', 'atan(v) (by el)'),
			]
		elif self.active_socket_set in 'By Vector':
			items = [
				# Vector -> Number
				('NORM_2', '||v||₂', 'norm(v, 2) (by Vec)'),
			]
		elif self.active_socket_set == 'By Matrix':
			items = [
				# Matrix -> Number
				('DET', 'det V', 'det(V) (by Mat)'),
				('COND', 'κ(V)', 'cond(V) (by Mat)'),
				('NORM_FRO', '||V||_F', 'norm(V, frobenius) (by Mat)'),
				('RANK', 'rank V', 'rank(V) (by Mat)'),
				None,
				# Matrix -> Array
				('DIAG', 'diag V', 'diag(V) (by Mat)'),
				('EIG_VALS', 'eigvals V', 'eigvals(V) (by Mat)'),
				('SVD_VALS', 'svdvals V', 'diag(svd(V)) (by Mat)'),
				None,
				# Matrix -> Matrix
				('INV', 'V⁻¹', 'V^(-1) (by Mat)'),
				('TRA', 'Vt', 'V^T (by Mat)'),
				None,
				# Matrix -> Matrices
				('QR', 'qr V', 'qr(V) -> Q·R (by Mat)'),
				('CHOL', 'chol V', 'cholesky(V) -> V·V† (by Mat)'),
				('SVD', 'svd V', 'svd(V) -> U·Σ·V† (by Mat)'),
			]
		elif self.active_socket_set == 'Expr':
			items = [('EXPR_EL', 'By Element', 'Expression-defined (by el)')]
		else:
			msg = f'Active socket set {self.active_socket_set} is unknown'
			raise RuntimeError(msg)

		return [
			(*item, '', i) if item is not None else None for i, item in enumerate(items)
		]

	def draw_props(self, _: bpy.types.Context, layout: bpy.types.UILayout) -> None:
		layout.prop(self, self.blfields['operation'], text='')

	####################
	# - Events
	####################
	@events.on_value_changed(
		prop_name='active_socket_set',
		run_on_init=True,
	)
	def on_socket_set_changed(self):
		self.operation = bl_cache.Signal.ResetEnumItems

	####################
	# - Compute: LazyValueFunc / Array
	####################
	@events.computes_output_socket(
		'Data',
		kind=ct.FlowKind.LazyValueFunc,
		props={'active_socket_set', 'operation'},
		input_sockets={'Data', 'Mapper'},
		input_socket_kinds={
			'Data': ct.FlowKind.LazyValueFunc,
			'Mapper': ct.FlowKind.LazyValueFunc,
		},
		input_sockets_optional={'Mapper': True},
	)
	def compute_data(self, props: dict, input_sockets: dict):
		has_data = not ct.FlowSignal.check(input_sockets['Data'])
		if (
			not has_data
			or props['operation'] == 'NONE'
			or (
				props['active_socket_set'] == 'Expr'
				and ct.FlowSignal.check(input_sockets['Mapper'])
			)
		):
			return ct.FlowSignal.FlowPending

		mapping_func: typ.Callable[[jax.Array], jax.Array] = {
			'By Element': {
				'REAL': lambda data: jnp.real(data),
				'IMAG': lambda data: jnp.imag(data),
				'ABS': lambda data: jnp.abs(data),
				'SQ': lambda data: jnp.square(data),
				'SQRT': lambda data: jnp.sqrt(data),
				'INV_SQRT': lambda data: 1 / jnp.sqrt(data),
				'COS': lambda data: jnp.cos(data),
				'SIN': lambda data: jnp.sin(data),
				'TAN': lambda data: jnp.tan(data),
				'ACOS': lambda data: jnp.acos(data),
				'ASIN': lambda data: jnp.asin(data),
				'ATAN': lambda data: jnp.atan(data),
				'SINC': lambda data: jnp.sinc(data),
			},
			'By Vector': {
				'NORM_2': lambda data: jnp.linalg.norm(data, ord=2, axis=-1),
			},
			'By Matrix': {
				# Matrix -> Number
				'DET': lambda data: jnp.linalg.det(data),
				'COND': lambda data: jnp.linalg.cond(data),
				'NORM_FRO': lambda data: jnp.linalg.matrix_norm(data, ord='fro'),
				'RANK': lambda data: jnp.linalg.matrix_rank(data),
				# Matrix -> Vec
				'DIAG': lambda data: jnp.diagonal(data, axis1=-2, axis2=-1),
				'EIG_VALS': lambda data: jnp.linalg.eigvals(data),
				'SVD_VALS': lambda data: jnp.linalg.svdvals(data),
				# Matrix -> Matrix
				'INV': lambda data: jnp.linalg.inv(data),
				'TRA': lambda data: jnp.matrix_transpose(data),
				# Matrix -> Matrices
				'QR': lambda data: jnp.linalg.qr(data),
				'CHOL': lambda data: jnp.linalg.cholesky(data),
				'SVD': lambda data: jnp.linalg.svd(data),
			},
			'Expr': {
				'EXPR_EL': lambda data: input_sockets['Mapper'].func(data),
			},
		}[props['active_socket_set']][props['operation']]

		# Compose w/Lazy Root Function Data
		return input_sockets['Data'].compose_within(
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

		if all(not ct.FlowSignal.check(inp) for inp in [lazy_value_func, params]):
			return ct.ArrayFlow(
				values=lazy_value_func.func_jax(
					*params.func_args, **params.func_kwargs
				),
				unit=None,
			)

		return ct.FlowSignal.FlowPending

	####################
	# - Compute Auxiliary: Info / Params
	####################
	@events.computes_output_socket(
		'Data',
		kind=ct.FlowKind.Info,
		props={'active_socket_set', 'operation'},
		input_sockets={'Data'},
		input_socket_kinds={'Data': ct.FlowKind.Info},
	)
	def compute_data_info(self, props: dict, input_sockets: dict) -> ct.InfoFlow:
		info = input_sockets['Data']
		if ct.FlowSignal.check(info):
			return ct.FlowSignal.FlowPending

		# Complex -> Real
		if props['active_socket_set'] == 'By Element' and props['operation'] in [
			'REAL',
			'IMAG',
			'ABS',
		]:
			return info.set_output_mathtype(spux.MathType.Real)

		if props['active_socket_set'] == 'By Vector' and props['operation'] in [
			'NORM_2'
		]:
			return {
				'NORM_2': lambda: info.collapse_output(
					collapsed_name=f'||{info.output_name}||₂',
					collapsed_mathtype=spux.MathType.Real,
					collapsed_unit=info.output_unit,
				)
			}[props['operation']]()

		if props['active_socket_set'] == 'By Matrix' and props['operation'] in [
			'DET',
			'COND',
			'NORM_FRO',
			'RANK',
		]:
			return {
				'DET': lambda: info.collapse_output(
					collapsed_name=f'det {info.output_name}',
					collapsed_mathtype=info.output_mathtype,
					collapsed_unit=info.output_unit,
				),
				'COND': lambda: info.collapse_output(
					collapsed_name=f'κ({info.output_name})',
					collapsed_mathtype=spux.MathType.Real,
					collapsed_unit=None,
				),
				'NORM_FRO': lambda: info.collapse_output(
					collapsed_name=f'||({info.output_name}||_F',
					collapsed_mathtype=spux.MathType.Real,
					collapsed_unit=info.output_unit,
				),
				'RANK': lambda: info.collapse_output(
					collapsed_name=f'rank {info.output_name}',
					collapsed_mathtype=spux.MathType.Integer,
					collapsed_unit=None,
				),
			}[props['operation']]()

		return info

	@events.computes_output_socket(
		'Data',
		kind=ct.FlowKind.Params,
		input_sockets={'Data'},
		input_socket_kinds={'Data': ct.FlowKind.Params},
	)
	def compute_data_params(self, input_sockets: dict) -> ct.ParamsFlow | ct.FlowSignal:
		return input_sockets['Data']


####################
# - Blender Registration
####################
BL_REGISTER = [
	MapMathNode,
]
BL_NODES = {ct.NodeType.MapMath: (ct.NodeCategory.MAXWELLSIM_ANALYSIS_MATH)}
