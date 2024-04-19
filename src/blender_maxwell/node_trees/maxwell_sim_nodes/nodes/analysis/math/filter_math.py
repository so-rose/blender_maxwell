import typing as typ

import bpy
import jax.numpy as jnp

from blender_maxwell.utils import logger

from .... import contracts as ct
from .... import sockets
from ... import base, events

log = logger.get(__name__)


# @functools.partial(jax.jit, static_argnames=('fixed_axis', 'fixed_axis_value'))
# jax.jit
def fix_axis(data, fixed_axis: int, fixed_axis_value: float):
	log.critical(data.shape)
	# Select Values of Fixed Axis
	fixed_axis_values = data[
		tuple(slice(None) if i == fixed_axis else 0 for i in range(data.ndim))
	]
	log.critical(fixed_axis_values)

	# Compute Nearest Index on Fixed Axis
	idx_of_nearest = jnp.argmin(jnp.abs(fixed_axis_values - fixed_axis_value))
	log.critical(idx_of_nearest)

	# Select Values along Fixed Axis Value
	return jnp.take(data, idx_of_nearest, axis=fixed_axis)


class FilterMathNode(base.MaxwellSimNode):
	node_type = ct.NodeType.FilterMath
	bl_label = 'Filter Math'

	input_sockets: typ.ClassVar = {
		'Data': sockets.AnySocketDef(),
	}
	input_socket_sets: typ.ClassVar = {
		'By Axis Value': {
			'Axis': sockets.IntegerNumberSocketDef(),
			'Value': sockets.RealNumberSocketDef(),
		},
		'By Axis': {
			'Axis': sockets.IntegerNumberSocketDef(),
		},
		## TODO: bool arrays for comparison/switching/sparse 0-setting/etc. .
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
		update=lambda self, context: self.on_prop_changed('operation', context),
	)

	def search_operations(self) -> list[tuple[str, str, str]]:
		items = []
		if self.active_socket_set == 'By Axis Value':
			items += [
				('FIX', 'Fix Coordinate', '(*, N, *) -> (*, *)'),
			]
		if self.active_socket_set == 'By Axis':
			items += [
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
		props={'operation', 'active_socket_set'},
		input_sockets={'Data', 'Axis', 'Value'},
		input_sockets_optional={'Axis': True, 'Value': True},
	)
	def compute_data(self, props: dict, input_sockets: dict):
		if not hasattr(input_sockets['Data'], 'shape'):
			msg = 'Input socket "Data" must be an N-D Array (with a "shape" attribute)'
			raise ValueError(msg)

		# By Axis Value
		if props['active_socket_set'] == 'By Axis Value':
			if props['operation'] == 'FIX':
				return fix_axis(
					input_sockets['Data'], input_sockets['Axis'], input_sockets['Value']
				)

		# By Axis
		if props['active_socket_set'] == 'By Axis':
			if props['operation'] == 'SQUEEZE':
				return jnp.squeeze(input_sockets['Data'], axis=input_sockets['Axis'])

		msg = 'Operation invalid'
		raise ValueError(msg)


####################
# - Blender Registration
####################
BL_REGISTER = [
	FilterMathNode,
]
BL_NODES = {ct.NodeType.FilterMath: (ct.NodeCategory.MAXWELLSIM_ANALYSIS_MATH)}
