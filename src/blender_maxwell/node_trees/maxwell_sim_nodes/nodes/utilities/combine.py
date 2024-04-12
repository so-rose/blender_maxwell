import typing as typ

import bpy
import sympy as sp

from ... import contracts as ct
from ... import sockets
from .. import base, events


class CombineNode(base.MaxwellSimNode):
	node_type = ct.NodeType.Combine
	bl_label = 'Combine'

	####################
	# - Sockets
	####################
	input_socket_sets: typ.ClassVar = {
		'Maxwell Sources': {},
		'Maxwell Structures': {},
		'Maxwell Monitors': {},
		'Real 3D Vector': {f'x_{i}': sockets.RealNumberSocketDef() for i in range(3)},
		# "Point 3D": {
		# axis: sockets.PhysicalLengthSocketDef()
		# for i, axis in zip(
		# range(3),
		# ["x", "y", "z"]
		# )
		# },
		# "Size 3D": {
		# axis_key: sockets.PhysicalLengthSocketDef()
		# for i, axis_key, axis_label in zip(
		# range(3),
		# ["x_size", "y_size", "z_size"],
		# ["X Size", "Y Size", "Z Size"],
		# )
		# },
	}
	output_socket_sets = {
		'Maxwell Sources': {
			'Sources': sockets.MaxwellSourceSocketDef(
				is_list=True,
			),
		},
		'Maxwell Structures': {
			'Structures': sockets.MaxwellStructureSocketDef(
				is_list=True,
			),
		},
		'Maxwell Monitors': {
			'Monitors': sockets.MaxwellMonitorSocketDef(
				is_list=True,
			),
		},
		'Real 3D Vector': {
			'Real 3D Vector': sockets.Real3DVectorSocketDef(),
		},
		# "Point 3D": {
		# "3D Point": sockets.PhysicalPoint3DSocketDef(),
		# },
		# "Size 3D": {
		# "3D Size": sockets.PhysicalSize3DSocketDef(),
		# },
	}

	amount: bpy.props.IntProperty(
		name='# Objects to Combine',
		description='Amount of Objects to Combine',
		default=1,
		min=1,
		# max=MAX_AMOUNT,
		update=lambda self, context: self.sync_prop('amount', context),
	)

	####################
	# - Draw
	####################
	def draw_props(self, context, layout):
		layout.prop(self, 'amount', text='#')

	####################
	# - Output Socket Computation
	####################
	@events.computes_output_socket(
		'Real 3D Vector', input_sockets={'x_0', 'x_1', 'x_2'}
	)
	def compute_real_3d_vector(self, input_sockets) -> sp.Expr:
		return sp.Matrix([input_sockets[f'x_{i}'] for i in range(3)])

	@events.computes_output_socket(
		'Sources',
		all_loose_input_sockets=True,
		props={'amount'},
	)
	def compute_sources(self, loose_input_sockets, props) -> sp.Expr:
		return [loose_input_sockets[f'Source #{i}'] for i in range(props['amount'])]

	@events.computes_output_socket(
		'Structures',
		all_loose_input_sockets=True,
		props={'amount'},
	)
	def compute_structures(self, loose_input_sockets, props) -> sp.Expr:
		return [loose_input_sockets[f'Structure #{i}'] for i in range(props['amount'])]

	@events.computes_output_socket(
		'Monitors',
		all_loose_input_sockets=True,
		props={'amount'},
	)
	def compute_monitors(self, loose_input_sockets, props) -> sp.Expr:
		return [loose_input_sockets[f'Monitor #{i}'] for i in range(props['amount'])]

	####################
	# - Input Socket Compilation
	####################
	@events.on_value_changed(
		prop_name='active_socket_set',
		props={'active_socket_set', 'amount'},
		run_on_init=True,
	)
	def on_value_changed__active_socket_set(self, props):
		if props['active_socket_set'] == 'Maxwell Sources':
			self.loose_input_sockets = {
				f'Source #{i}': sockets.MaxwellSourceSocketDef()
				for i in range(props['amount'])
			}
		elif props['active_socket_set'] == 'Maxwell Structures':
			self.loose_input_sockets = {
				f'Structure #{i}': sockets.MaxwellStructureSocketDef()
				for i in range(props['amount'])
			}
		elif props['active_socket_set'] == 'Maxwell Monitors':
			self.loose_input_sockets = {
				f'Monitor #{i}': sockets.MaxwellMonitorSocketDef()
				for i in range(props['amount'])
			}
		else:
			self.loose_input_sockets = {}

	@events.on_value_changed(
		prop_name='amount',
	)
	def on_value_changed__amount(self):
		self.on_value_changed__active_socket_set()


####################
# - Blender Registration
####################
BL_REGISTER = [
	CombineNode,
]
BL_NODES = {ct.NodeType.Combine: (ct.NodeCategory.MAXWELLSIM_UTILITIES)}
