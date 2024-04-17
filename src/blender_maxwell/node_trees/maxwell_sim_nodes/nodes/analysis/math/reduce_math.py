import typing as typ

import bpy
import jax.numpy as jnp
import sympy as sp

from blender_maxwell.utils import logger

from .... import contracts as ct
from .... import sockets
from ... import base, events

log = logger.get(__name__)


class ReduceMathNode(base.MaxwellSimNode):
	node_type = ct.NodeType.ReduceMath
	bl_label = 'Reduce Math'

	input_sockets: typ.ClassVar = {
		'Data': sockets.AnySocketDef(),
		'Axis': sockets.IntegerNumberSocketDef(),
	}
	input_socket_sets: typ.ClassVar = {
		'By Axis': {
			'Axis': sockets.IntegerNumberSocketDef(),
		},
		'Expr': {
			'Reducer': sockets.ExprSocketDef(
				symbols=[sp.Symbol('a'), sp.Symbol('b')],
				default_expr=sp.Symbol('a') + sp.Symbol('b'),
			),
			'Axis': sockets.IntegerNumberSocketDef(),
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
		description='Operation to reduce the input axis with',
		items=lambda self, _: self.search_operations(),
		update=lambda self, context: self.sync_prop('operation', context),
	)

	def search_operations(self) -> list[tuple[str, str, str]]:
		items = []
		if self.active_socket_set == 'By Axis':
			items += [
				# Accumulation
				('SUM', 'Sum', 'sum(*, N, *) -> (*, 1, *)'),
				('PROD', 'Prod', 'prod(*, N, *) -> (*, 1, *)'),
				('MIN', 'Axis-Min', '(*, N, *) -> (*, 1, *)'),
				('MAX', 'Axis-Max', '(*, N, *) -> (*, 1, *)'),
				('P2P', 'Peak-to-Peak', '(*, N, *) -> (*, 1 *)'),
				# Stats
				('MEAN', 'Mean', 'mean(*, N, *) -> (*, 1, *)'),
				('MEDIAN', 'Median', 'median(*, N, *) -> (*, 1, *)'),
				('STDDEV', 'Std Dev', 'stddev(*, N, *) -> (*, 1, *)'),
				('VARIANCE', 'Variance', 'var(*, N, *) -> (*, 1, *)'),
				# Dimension Reduction
				('SQUEEZE', 'Squeeze', '(*, 1, *) -> (*, *)'),
			]
		else:
			items += [('NONE', 'None', 'No operations...')]

		return items

	def draw_props(self, _: bpy.types.Context, layout: bpy.types.UILayout) -> None:
		if self.active_socket_set != 'Axis Expr':
			layout.prop(self, 'operation')

	####################
	# - Compute
	####################
	@events.computes_output_socket(
		'Data',
		props={'operation'},
		input_sockets={'Data', 'Axis', 'Reducer'},
		input_socket_kinds={'Reducer': ct.DataFlowKind.LazyValue},
		input_sockets_optional={'Reducer': True},
	)
	def compute_data(self, props: dict, input_sockets: dict):
		if not hasattr(input_sockets['Data'], 'shape'):
			msg = 'Input socket "Data" must be an N-D Array (with a "shape" attribute)'
			raise ValueError(msg)

		if self.active_socket_set == 'Axis Expr':
			ufunc = jnp.ufunc(input_sockets['Reducer'], nin=2, nout=1)
			return ufunc.reduce(input_sockets['Data'], axis=input_sockets['Axis'])

		if self.active_socket_set == 'By Axis':
			## Dimension Reduction
			# ('SQUEEZE', 'Squeeze', '(*, 1, *) -> (*, *)'),
			# Accumulation
			if props['operation'] == 'SUM':
				return jnp.sum(input_sockets['Data'], axis=input_sockets['Axis'])
			if props['operation'] == 'PROD':
				return jnp.prod(input_sockets['Data'], axis=input_sockets['Axis'])
			if props['operation'] == 'MIN':
				return jnp.min(input_sockets['Data'], axis=input_sockets['Axis'])
			if props['operation'] == 'MAX':
				return jnp.max(input_sockets['Data'], axis=input_sockets['Axis'])
			if props['operation'] == 'P2P':
				return jnp.p2p(input_sockets['Data'], axis=input_sockets['Axis'])

			# Stats
			if props['operation'] == 'MEAN':
				return jnp.mean(input_sockets['Data'], axis=input_sockets['Axis'])
			if props['operation'] == 'MEDIAN':
				return jnp.median(input_sockets['Data'], axis=input_sockets['Axis'])
			if props['operation'] == 'STDDEV':
				return jnp.std(input_sockets['Data'], axis=input_sockets['Axis'])
			if props['operation'] == 'VARIANCE':
				return jnp.var(input_sockets['Data'], axis=input_sockets['Axis'])

			# Dimension Reduction
			if props['operation'] == 'SQUEEZE':
				return jnp.squeeze(input_sockets['Data'], axis=input_sockets['Axis'])

		msg = 'Operation invalid'
		raise ValueError(msg)


####################
# - Blender Registration
####################
BL_REGISTER = [
	ReduceMathNode,
]
BL_NODES = {ct.NodeType.ReduceMath: (ct.NodeCategory.MAXWELLSIM_ANALYSIS_MATH)}
