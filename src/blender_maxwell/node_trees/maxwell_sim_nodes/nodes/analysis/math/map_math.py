"""Declares `MapMathNode`."""

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

X_COMPLEX = sp.Symbol('x', complex=True)


class MapOperation(enum.StrEnum):
	"""Valid operations for the `MapMathNode`.

	Attributes:
		UserExpr: Use a user-provided mapping expression.
		Real: Compute the real part of the input.
		Imag: Compute the imaginary part of the input.
		Abs: Compute the absolute value of the input.
		Sq: Square the input.
		Sqrt: Compute the (principal) square root of the input.
		InvSqrt: Compute the inverse square root of the input.
		Cos: Compute the cosine of the input.
		Sin: Compute the sine of the input.
		Tan: Compute the tangent of the input.
		Acos: Compute the inverse cosine of the input.
		Asin: Compute the inverse sine of the input.
		Atan: Compute the inverse tangent of the input.
		Norm2: Compute the 2-norm (aka. length) of the input vector.
		Det: Compute the determinant of the input matrix.
		Cond: Compute the condition number of the input matrix.
		NormFro: Compute the frobenius norm of the input matrix.
		Rank: Compute the rank of the input matrix.
		Diag: Compute the diagonal vector of the input matrix.
		EigVals: Compute the eigenvalues vector of the input matrix.
		SvdVals: Compute the singular values vector of the input matrix.
		Inv: Compute the inverse matrix of the input matrix.
		Tra: Compute the transpose matrix of the input matrix.
		Qr: Compute the QR-factorized matrices of the input matrix.
		Chol: Compute the Cholesky-factorized matrices of the input matrix.
		Svd: Compute the SVD-factorized matrices of the input matrix.
	"""

	# By User Expression
	UserExpr = enum.auto()
	# By Number
	Real = enum.auto()
	Imag = enum.auto()
	Abs = enum.auto()
	Sq = enum.auto()
	Sqrt = enum.auto()
	InvSqrt = enum.auto()
	Cos = enum.auto()
	Sin = enum.auto()
	Tan = enum.auto()
	Acos = enum.auto()
	Asin = enum.auto()
	Atan = enum.auto()
	Sinc = enum.auto()
	# By Vector
	Norm2 = enum.auto()
	# By Matrix
	Det = enum.auto()
	Cond = enum.auto()
	NormFro = enum.auto()
	Rank = enum.auto()
	Diag = enum.auto()
	EigVals = enum.auto()
	SvdVals = enum.auto()
	Inv = enum.auto()
	Tra = enum.auto()
	Qr = enum.auto()
	Chol = enum.auto()
	Svd = enum.auto()

	@staticmethod
	def to_name(value: typ.Self) -> str:
		MO = MapOperation
		return {
			# By User Expression
			MO.UserExpr: '*',
			# By Number
			MO.Real: 'ℝ(v)',
			MO.Imag: 'Im(v)',
			MO.Abs: '|v|',
			MO.Sq: 'v²',
			MO.Sqrt: '√v',
			MO.InvSqrt: '1/√v',
			MO.Cos: 'cos v',
			MO.Sin: 'sin v',
			MO.Tan: 'tan v',
			MO.Acos: 'acos v',
			MO.Asin: 'asin v',
			MO.Atan: 'atan v',
			MO.Sinc: 'sinc v',
			# By Vector
			MO.Norm2: '||v||₂',
			# By Matrix
			MO.Det: 'det V',
			MO.Cond: 'κ(V)',
			MO.NormFro: '||V||_F',
			MO.Rank: 'rank V',
			MO.Diag: 'diag V',
			MO.EigVals: 'eigvals V',
			MO.SvdVals: 'svdvals V',
			MO.Inv: 'V⁻¹',
			MO.Tra: 'Vt',
			MO.Qr: 'qr V',
			MO.Chol: 'chol V',
			MO.Svd: 'svd V',
		}[value]

	@staticmethod
	def to_icon(value: typ.Self) -> str:
		return ''

	def bl_enum_element(self, i: int) -> ct.BLEnumElement:
		MO = MapOperation
		return (
			str(self),
			MO.to_name(self),
			MO.to_name(self),
			MO.to_icon(self),
			i,
		)

	@staticmethod
	def by_element_shape(shape: tuple[int, ...] | None) -> list[typ.Self]:
		MO = MapOperation
		if shape == 'noshape':
			return []

		# By Number
		if shape is None:
			return [
				MO.Real,
				MO.Imag,
				MO.Abs,
				MO.Sq,
				MO.Sqrt,
				MO.InvSqrt,
				MO.Cos,
				MO.Sin,
				MO.Tan,
				MO.Acos,
				MO.Asin,
				MO.Atan,
				MO.Sinc,
			]

		# By Vector
		if len(shape) == 1:
			return [
				MO.Norm2,
			]
		# By Matrix
		if len(shape) == 2:
			return [
				MO.Det,
				MO.Cond,
				MO.NormFro,
				MO.Rank,
				MO.Diag,
				MO.EigVals,
				MO.SvdVals,
				MO.Inv,
				MO.Tra,
				MO.Qr,
				MO.Chol,
				MO.Svd,
			]

		return []

	def jax_func(self, user_expr_func: ct.LazyValueFuncFlow | None = None):
		MO = MapOperation
		if self == MO.UserExpr and user_expr_func is not None:
			return lambda data: user_expr_func.func(data)
		return {
			# By Number
			MO.Real: lambda data: jnp.real(data),
			MO.Imag: lambda data: jnp.imag(data),
			MO.Abs: lambda data: jnp.abs(data),
			MO.Sq: lambda data: jnp.square(data),
			MO.Sqrt: lambda data: jnp.sqrt(data),
			MO.InvSqrt: lambda data: 1 / jnp.sqrt(data),
			MO.Cos: lambda data: jnp.cos(data),
			MO.Sin: lambda data: jnp.sin(data),
			MO.Tan: lambda data: jnp.tan(data),
			MO.Acos: lambda data: jnp.acos(data),
			MO.Asin: lambda data: jnp.asin(data),
			MO.Atan: lambda data: jnp.atan(data),
			MO.Sinc: lambda data: jnp.sinc(data),
			# By Vector
			# Vector -> Number
			MO.Norm2: lambda data: jnp.linalg.norm(data, ord=2, axis=-1),
			# By Matrix
			# Matrix -> Number
			MO.Det: lambda data: jnp.linalg.det(data),
			MO.Cond: lambda data: jnp.linalg.cond(data),
			MO.NormFro: lambda data: jnp.linalg.matrix_norm(data, ord='fro'),
			MO.Rank: lambda data: jnp.linalg.matrix_rank(data),
			# Matrix -> Vec
			MO.Diag: lambda data: jnp.diagonal(data, axis1=-2, axis2=-1),
			MO.EigVals: lambda data: jnp.linalg.eigvals(data),
			MO.SvdVals: lambda data: jnp.linalg.svdvals(data),
			# Matrix -> Matrix
			MO.Inv: lambda data: jnp.linalg.inv(data),
			MO.Tra: lambda data: jnp.matrix_transpose(data),
			# Matrix -> Matrices
			MO.Qr: lambda data: jnp.linalg.qr(data),
			MO.Chol: lambda data: jnp.linalg.cholesky(data),
			MO.Svd: lambda data: jnp.linalg.svd(data),
		}[self]

	def transform_info(self, info: ct.InfoFlow):
		MO = MapOperation
		return {
			# By User Expression
			MO.UserExpr: '*',
			# By Number
			MO.Real: lambda: info.set_output_mathtype(spux.MathType.Real),
			MO.Imag: lambda: info.set_output_mathtype(spux.MathType.Real),
			MO.Abs: lambda: info.set_output_mathtype(spux.MathType.Real),
			# By Vector
			MO.Norm2: lambda: info.collapse_output(
				collapsed_name=MO.to_name(self).replace('v', info.output_name),
				collapsed_mathtype=spux.MathType.Real,
				collapsed_unit=info.output_unit,
			),
			# By Matrix
			MO.Det: lambda: info.collapse_output(
				collapsed_name=MO.to_name(self).replace('V', info.output_name),
				collapsed_mathtype=info.output_mathtype,
				collapsed_unit=info.output_unit,
			),
			MO.Cond: lambda: info.collapse_output(
				collapsed_name=MO.to_name(self).replace('V', info.output_name),
				collapsed_mathtype=spux.MathType.Real,
				collapsed_unit=None,
			),
			MO.NormFro: lambda: info.collapse_output(
				collapsed_name=MO.to_name(self).replace('V', info.output_name),
				collapsed_mathtype=spux.MathType.Real,
				collapsed_unit=info.output_unit,
			),
			MO.Rank: lambda: info.collapse_output(
				collapsed_name=MO.to_name(self).replace('V', info.output_name),
				collapsed_mathtype=spux.MathType.Integer,
				collapsed_unit=None,
			),
			## TODO: Matrix -> Vec
			## TODO: Matrix -> Matrices
		}.get(self, info)()


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
		'Expr': sockets.ExprSocketDef(active_kind=ct.FlowKind.Array),
	}
	output_sockets: typ.ClassVar = {
		'Expr': sockets.ExprSocketDef(active_kind=ct.FlowKind.Array),
	}

	####################
	# - Properties
	####################
	operation: MapOperation = bl_cache.BLField(
		prop_ui=True, enum_cb=lambda self, _: self.search_operations()
	)

	@property
	def expr_output_shape(self) -> ct.InfoFlow | None:
		info = self._compute_input('Expr', kind=ct.FlowKind.Info)
		has_info = not ct.FlowSignal.check(info)
		if has_info:
			return info.output_shape

		return 'noshape'

	output_shape: tuple[int, ...] | None = bl_cache.BLField(None)

	def search_operations(self) -> list[ct.BLEnumElement]:
		if self.expr_output_shape != 'noshape':
			return [
				operation.bl_enum_element(i)
				for i, operation in enumerate(
					MapOperation.by_element_shape(self.expr_output_shape)
				)
			]
		return []

	####################
	# - UI
	####################
	def draw_label(self):
		if self.operation is not None:
			return 'Map: ' + MapOperation.to_name(self.operation)

		return self.bl_label

	def draw_props(self, _: bpy.types.Context, layout: bpy.types.UILayout) -> None:
		layout.prop(self, self.blfields['operation'], text='')

	####################
	# - Events
	####################
	@events.on_value_changed(
		# Trigger
		socket_name='Expr',
		run_on_init=True,
	)
	def on_input_changed(self):
		if self.operation not in MapOperation.by_element_shape(self.expr_output_shape):
			self.operation = bl_cache.Signal.ResetEnumItems

	@events.on_value_changed(
		# Trigger
		prop_name={'operation'},
		run_on_init=True,
		# Loaded
		props={'operation'},
	)
	def on_operation_changed(self, props: dict) -> None:
		operation = props['operation']

		# UserExpr: Add/Remove Input Socket
		if operation == MapOperation.UserExpr:
			current_bl_socket = self.loose_input_sockets.get('Mapper')
			if current_bl_socket is None:
				self.loose_input_sockets = {
					'Mapper': sockets.ExprSocketDef(
						symbols={X_COMPLEX},
						default_value=X_COMPLEX,
						mathtype=spux.MathType.Complex,
					),
				}

		elif self.loose_input_sockets:
			self.loose_input_sockets = {}

	####################
	# - Compute: LazyValueFunc / Array
	####################
	@events.computes_output_socket(
		'Expr',
		kind=ct.FlowKind.LazyValueFunc,
		props={'operation'},
		input_sockets={'Expr', 'Mapper'},
		input_socket_kinds={
			'Expr': ct.FlowKind.LazyValueFunc,
			'Mapper': ct.FlowKind.LazyValueFunc,
		},
		input_sockets_optional={'Mapper': True},
	)
	def compute_data(self, props: dict, input_sockets: dict):
		operation = props['operation']
		expr = input_sockets['Expr']
		mapper = input_sockets['Mapper']

		has_expr = not ct.FlowSignal.check(expr)
		has_mapper = not ct.FlowSignal.check(mapper)

		if has_expr and operation is not None:
			if not has_mapper:
				return expr.compose_within(
					operation.jax_func(),
					supports_jax=True,
				)
			if operation == MapOperation.UserExpr and has_mapper:
				return expr.compose_within(
					operation.jax_func(user_expr_func=mapper),
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
			)
		return ct.FlowSignal.FlowPending

	####################
	# - Compute Auxiliary: Info / Params
	####################
	@events.computes_output_socket(
		'Expr',
		kind=ct.FlowKind.Info,
		props={'active_socket_set', 'operation'},
		input_sockets={'Expr'},
		input_socket_kinds={'Expr': ct.FlowKind.Info},
	)
	def compute_data_info(self, props: dict, input_sockets: dict) -> ct.InfoFlow:
		operation = props['operation']
		info = input_sockets['Expr']

		has_info = not ct.FlowSignal.check(info)

		if has_info and operation is not None:
			return operation.transform_info(info)

		return ct.FlowSignal.FlowPending

	@events.computes_output_socket(
		'Expr',
		kind=ct.FlowKind.Params,
		input_sockets={'Expr'},
		input_socket_kinds={'Expr': ct.FlowKind.Params},
	)
	def compute_data_params(self, input_sockets: dict) -> ct.ParamsFlow | ct.FlowSignal:
		return input_sockets['Expr']


####################
# - Blender Registration
####################
BL_REGISTER = [
	MapMathNode,
]
BL_NODES = {ct.NodeType.MapMath: (ct.NodeCategory.MAXWELLSIM_ANALYSIS_MATH)}
