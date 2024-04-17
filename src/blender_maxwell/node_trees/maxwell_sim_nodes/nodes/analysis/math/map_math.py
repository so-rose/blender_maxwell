import typing as typ

import bpy
import jax
import jax.numpy as jnp
import sympy as sp

from blender_maxwell.utils import logger

from .... import contracts as ct
from .... import sockets
from ... import base, events

log = logger.get(__name__)


class MapMathNode(base.MaxwellSimNode):
	node_type = ct.NodeType.MapMath
	bl_label = 'Map Math'

	input_sockets: typ.ClassVar = {
		'Data': sockets.AnySocketDef(),
	}
	input_socket_sets: typ.ClassVar = {
		'By Element': {},
		'By Vector': {},
		'By Matrix': {},
		'Expr': {
			'Mapper': sockets.ExprSocketDef(
				symbols=[sp.Symbol('x')],
				default_expr=sp.Symbol('x'),
			),
		},
	}
	output_sockets: typ.ClassVar = {
		'Data': sockets.AnySocketDef(),
	}

	####################
	# - Properties
	####################
	operation: bpy.props.EnumProperty(
		name='Op',
		description='Operation to apply to the input',
		items=lambda self, _: self.search_operations(),
		update=lambda self, context: self.sync_prop('operation', context),
	)

	def search_operations(self) -> list[tuple[str, str, str]]:
		items = []
		if self.active_socket_set == 'By Element':
			items += [
				# General
				('REAL', 'real', 'ℝ(L) (by el)'),
				('IMAG', 'imag', 'Im(L) (by el)'),
				('ABS', 'abs', '|L| (by el)'),
				('SQ', 'square', 'L^2 (by el)'),
				('SQRT', 'sqrt', 'sqrt(L) (by el)'),
				('INV_SQRT', '1/sqrt', '1/sqrt(L) (by el)'),
				# Trigonometry
				('COS', 'cos', 'cos(L) (by el)'),
				('SIN', 'sin', 'sin(L) (by el)'),
				('TAN', 'tan', 'tan(L) (by el)'),
				('ACOS', 'acos', 'acos(L) (by el)'),
				('ASIN', 'asin', 'asin(L) (by el)'),
				('ATAN', 'atan', 'atan(L) (by el)'),
			]
		elif self.active_socket_set in 'By Vector':
			items += [
				('NORM_2', '2-Norm', '||L||_2 (by Vec)'),
			]
		elif self.active_socket_set == 'By Matrix':
			items += [
				# Matrix -> Number
				('DET', 'Determinant', 'det(L) (by Mat)'),
				('COND', 'Condition', 'κ(L) (by Mat)'),
				('NORM_FRO', 'Frobenius Norm', '||L||_F (by Mat)'),
				('RANK', 'Rank', 'rank(L) (by Mat)'),
				# Matrix -> Array
				('DIAG', 'Diagonal', 'diag(L) (by Mat)'),
				('EIG_VALS', 'Eigenvalues', 'eigvals(L) (by Mat)'),
				('SVD_VALS', 'SVD', 'svd(L) -> diag(Σ) (by Mat)'),
				# Matrix -> Matrix
				('INV', 'Invert', 'L^(-1) (by Mat)'),
				('TRA', 'Transpose', 'L^T (by Mat)'),
				# Matrix -> Matrices
				('QR', 'QR', 'L -> Q·R (by Mat)'),
				('CHOL', 'Cholesky', 'L -> L·Lh (by Mat)'),
				('SVD', 'SVD', 'L -> U·Σ·Vh (by Mat)'),
			]
		else:
			items += ['EXPR_EL', 'Expr (by el)', 'Expression-defined (by el)']
		return items

	def draw_props(self, _: bpy.types.Context, layout: bpy.types.UILayout) -> None:
		if self.active_socket_set not in {'Expr (Element)'}:
			layout.prop(self, 'operation')

	####################
	# - Compute
	####################
	@events.computes_output_socket(
		'Data',
		props={'active_socket_set', 'operation'},
		input_sockets={'Data', 'Mapper'},
		input_socket_kinds={'Mapper': ct.DataFlowKind.LazyValue},
		input_sockets_optional={'Mapper': True},
	)
	def compute_data(self, props: dict, input_sockets: dict):
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
				'NORM_2': lambda data: jnp.norm(data, ord=2, axis=-1),
			},
			'By Matrix': {
				# Matrix -> Number
				'DET': lambda data: jnp.linalg.det(data),
				'COND': lambda data: jnp.linalg.cond(data),
				'NORM_FRO': lambda data: jnp.linalg.matrix_norm(data, ord='fro'),
				'RANK': lambda data: jnp.linalg.matrix_rank(data),
				# Matrix -> Vec
				'DIAG': lambda data: jnp.diag(data),
				'EIG_VALS': lambda data: jnp.eigvals(data),
				'SVD_VALS': lambda data: jnp.svdvals(data),
				# Matrix -> Matrix
				'INV': lambda data: jnp.inv(data),
				'TRA': lambda data: jnp.matrix_transpose(data),
				# Matrix -> Matrices
				'QR': lambda data: jnp.inv(data),
				'CHOL': lambda data: jnp.linalg.cholesky(data),
				'SVD': lambda data: jnp.linalg.svd(data),
			},
			'By El (Expr)': {
				'EXPR_EL': lambda data: input_sockets['Mapper'](data),
			},
		}[props['active_socket_set']][props['operation']]

		# Compose w/Lazy Root Function Data
		return input_sockets['Data'].compose(
			function=mapping_func,
		)


####################
# - Blender Registration
####################
BL_REGISTER = [
	MapMathNode,
]
BL_NODES = {ct.NodeType.MapMath: (ct.NodeCategory.MAXWELLSIM_ANALYSIS_MATH)}
