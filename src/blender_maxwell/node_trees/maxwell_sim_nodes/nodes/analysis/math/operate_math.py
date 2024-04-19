import typing as typ

import bpy
import jax.numpy as jnp

from blender_maxwell.utils import logger

from .... import contracts as ct
from .... import sockets
from ... import base, events

log = logger.get(__name__)


class OperateMathNode(base.MaxwellSimNode):
	node_type = ct.NodeType.OperateMath
	bl_label = 'Operate Math'

	input_socket_sets: typ.ClassVar = {
		'Elementwise': {
			'Data L': sockets.AnySocketDef(),
			'Data R': sockets.AnySocketDef(),
		},
		## TODO: Filter-array building operations
		'Vec-Vec': {
			'Data L': sockets.AnySocketDef(),
			'Data R': sockets.AnySocketDef(),
		},
		'Mat-Vec': {
			'Data L': sockets.AnySocketDef(),
			'Data R': sockets.AnySocketDef(),
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
		description='Operation to apply to the two inputs',
		items=lambda self, _: self.search_operations(),
		update=lambda self, context: self.on_prop_changed('operation', context),
	)

	def search_operations(self) -> list[tuple[str, str, str]]:
		items = []
		if self.active_socket_set == 'Elementwise':
			items = [
				('ADD', 'Add', 'L + R (by el)'),
				('SUB', 'Subtract', 'L - R (by el)'),
				('MUL', 'Multiply', 'L · R (by el)'),
				('DIV', 'Divide', 'L ÷ R (by el)'),
				('POW', 'Power', 'L^R (by el)'),
				('FMOD', 'Trunc Modulo', 'fmod(L,R) (by el)'),
				('ATAN2', 'atan2', 'atan2(L,R) (by el)'),
				('HEAVISIDE', 'Heaviside', '{0|L<0  1|L>0  R|L=0} (by el)'),
			]
		elif self.active_socket_set in 'Vec | Vec':
			items = [
				('DOT', 'Dot', 'L · R'),
				('CROSS', 'Cross', 'L x R (by last-axis'),
			]
		elif self.active_socket_set == 'Mat | Vec':
			items = [
				('DOT', 'Dot', 'L · R'),
				('LIN_SOLVE', 'Lin Solve', 'Lx = R -> x (by last-axis of R)'),
				('LSQ_SOLVE', 'LSq Solve', 'Lx = R ~> x (by last-axis of R)'),
			]
		return items

	def draw_props(self, _: bpy.types.Context, layout: bpy.types.UILayout) -> None:
		layout.prop(self, 'operation')

	####################
	# - Properties
	####################
	@events.computes_output_socket(
		'Data',
		props={'operation'},
		input_sockets={'Data L', 'Data R'},
	)
	def compute_data(self, props: dict, input_sockets: dict):
		if self.active_socket_set == 'Elementwise':
			# Element-Wise Arithmetic
			if props['operation'] == 'ADD':
				return input_sockets['Data L'] + input_sockets['Data R']
			if props['operation'] == 'SUB':
				return input_sockets['Data L'] - input_sockets['Data R']
			if props['operation'] == 'MUL':
				return input_sockets['Data L'] * input_sockets['Data R']
			if props['operation'] == 'DIV':
				return input_sockets['Data L'] / input_sockets['Data R']

			# Element-Wise Arithmetic
			if props['operation'] == 'POW':
				return input_sockets['Data L'] ** input_sockets['Data R']

			# Binary Trigonometry
			if props['operation'] == 'ATAN2':
				return jnp.atan2(input_sockets['Data L'], input_sockets['Data R'])

			# Special Functions
			if props['operation'] == 'HEAVISIDE':
				return jnp.heaviside(input_sockets['Data L'], input_sockets['Data R'])

		# Linear Algebra
		if self.active_socket_set in {'Vec-Vec', 'Mat-Vec'}:
			if props['operation'] == 'DOT':
				return jnp.dot(input_sockets['Data L'], input_sockets['Data R'])

		elif self.active_socket_set == 'Vec-Vec':
			if props['operation'] == 'CROSS':
				return jnp.cross(input_sockets['Data L'], input_sockets['Data R'])

		elif self.active_socket_set == 'Mat-Vec':
			if props['operation'] == 'LIN_SOLVE':
				return jnp.linalg.lstsq(
					input_sockets['Data L'], input_sockets['Data R']
				)
			if props['operation'] == 'LSQ_SOLVE':
				return jnp.linalg.solve(
					input_sockets['Data L'], input_sockets['Data R']
				)

		msg = 'Invalid operation'
		raise ValueError(msg)


####################
# - Blender Registration
####################
BL_REGISTER = [
	OperateMathNode,
]
BL_NODES = {ct.NodeType.OperateMath: (ct.NodeCategory.MAXWELLSIM_ANALYSIS_MATH)}
