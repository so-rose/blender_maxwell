"""Declares `FilterMathNode`."""

import enum
import typing as typ

import bpy
import jax
import jax.numpy as jnp

from blender_maxwell.utils import bl_cache, logger

from .... import contracts as ct
from .... import sockets
from ... import base, events

log = logger.get(__name__)


class FilterMathNode(base.MaxwellSimNode):
	r"""Applies a function that operates on the shape of the array.

	The shape, type, and interpretation of the input/output data is dynamically shown.

	# Socket Sets
	## Dimensions
	Alter the dimensions of the array.

	## Interpret
	Only alter the interpretation of the array data, which guides what it can be used for.

	These operations are **zero cost**, since the data itself is untouched.

	Attributes:
		operation: Operation to apply to the input.
	"""

	node_type = ct.NodeType.FilterMath
	bl_label = 'Filter Math'

	input_sockets: typ.ClassVar = {
		'Data': sockets.DataSocketDef(format='jax'),
	}
	input_socket_sets: typ.ClassVar = {
		'Interpret': {},
		'Dimensions': {},
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

	# Dimension Selection
	dim_0: enum.Enum = bl_cache.BLField(
		None, prop_ui=True, enum_cb=lambda self, _: self.search_dims()
	)
	dim_1: enum.Enum = bl_cache.BLField(
		None, prop_ui=True, enum_cb=lambda self, _: self.search_dims()
	)

	####################
	# - Computed
	####################
	@property
	def data_info(self) -> ct.InfoFlow | None:
		info = self._compute_input('Data', kind=ct.FlowKind.Info)
		if not ct.FlowSignal.check(info):
			return info

		return None

	####################
	# - Operation Search
	####################
	def search_operations(self) -> list[tuple[str, str, str]]:
		items = []
		if self.active_socket_set == 'Interpret':
			items += [
				('DIM_TO_VEC', '→ Vector', 'Shift last dimension to output.'),
				('DIMS_TO_MAT', '→ Matrix', 'Shift last 2 dimensions to output.'),
			]
		elif self.active_socket_set == 'Dimensions':
			items += [
				('PIN_LEN_ONE', 'pinₐ =1', 'Remove a len(1) dimension'),
				(
					'PIN',
					'pinₐ ≈v',
					'Remove a len(n) dimension by selecting an index',
				),
				('SWAP', 'a₁ ↔ a₂', 'Swap the position of two dimensions'),
			]

		return [(*item, '', i) for i, item in enumerate(items)]

	####################
	# - Dimensions Search
	####################
	def search_dims(self) -> list[ct.BLEnumElement]:
		if self.data_info is None:
			return []

		if self.operation == 'PIN_LEN_ONE':
			dims = [
				(dim_name, dim_name, f'Dimension "{dim_name}" of length 1')
				for dim_name in self.data_info.dim_names
				if self.data_info.dim_lens[dim_name] == 1
			]
		elif self.operation in ['PIN', 'SWAP']:
			dims = [
				(dim_name, dim_name, f'Dimension "{dim_name}"')
				for dim_name in self.data_info.dim_names
			]
		else:
			return []

		return [(*dim, '', i) for i, dim in enumerate(dims)]

	####################
	# - UI
	####################
	def draw_label(self):
		labels = {
			'PIN_LEN_ONE': lambda: f'Filter: Pin {self.dim_0} (len=1)',
			'PIN': lambda: f'Filter: Pin {self.dim_0}',
			'SWAP': lambda: f'Filter: Swap {self.dim_0}|{self.dim_1}',
			'DIM_TO_VEC': lambda: 'Filter: -> Vector',
			'DIMS_TO_MAT': lambda: 'Filter: -> Matrix',
		}

		if (label := labels.get(self.operation)) is not None:
			return label()

		return self.bl_label

	def draw_props(self, _: bpy.types.Context, layout: bpy.types.UILayout) -> None:
		layout.prop(self, self.blfields['operation'], text='')

		if self.active_socket_set == 'Dimensions':
			if self.operation in ['PIN_LEN_ONE', 'PIN']:
				layout.prop(self, self.blfields['dim_0'], text='')

			if self.operation == 'SWAP':
				row = layout.row(align=True)
				row.prop(self, self.blfields['dim_0'], text='')
				row.prop(self, self.blfields['dim_1'], text='')

	####################
	# - Events
	####################
	@events.on_value_changed(
		prop_name='active_socket_set',
		run_on_init=True,
	)
	def on_socket_set_changed(self):
		self.operation = bl_cache.Signal.ResetEnumItems

	@events.on_value_changed(
		# Trigger
		socket_name='Data',
		prop_name={'active_socket_set', 'operation'},
		run_on_init=True,
		# Loaded
		props={'operation'},
	)
	def on_any_change(self, props: dict) -> None:
		self.dim_0 = bl_cache.Signal.ResetEnumItems
		self.dim_1 = bl_cache.Signal.ResetEnumItems

	@events.on_value_changed(
		socket_name='Data',
		prop_name={'dim_0', 'dim_1', 'operation'},
		## run_on_init: Implicitly triggered.
		props={'operation', 'dim_0', 'dim_1'},
		input_sockets={'Data'},
		input_socket_kinds={'Data': ct.FlowKind.Info},
	)
	def on_dim_change(self, props: dict, input_sockets: dict):
		has_data = not ct.FlowSignal.check(input_sockets['Data'])
		if not has_data:
			return

		# "Dimensions"|"PIN": Add/Remove Input Socket
		if props['operation'] == 'PIN' and props['dim_0'] != 'NONE':
			# Get Current and Wanted Socket Defs
			current_bl_socket = self.loose_input_sockets.get('Value')
			wanted_socket_def = sockets.SOCKET_DEFS[
				ct.unit_to_socket_type(
					input_sockets['Data'].dim_idx[props['dim_0']].unit
				)
			]

			# Determine Whether to Declare New Loose Input SOcket
			if (
				current_bl_socket is None
				or sockets.SOCKET_DEFS[current_bl_socket.socket_type]
				!= wanted_socket_def
			):
				self.loose_input_sockets = {
					'Value': wanted_socket_def(),
				}
		elif self.loose_input_sockets:
			self.loose_input_sockets = {}

	####################
	# - Compute: LazyValueFunc / Array
	####################
	@events.computes_output_socket(
		'Data',
		kind=ct.FlowKind.LazyValueFunc,
		props={'operation', 'dim_0', 'dim_1'},
		input_sockets={'Data'},
		input_socket_kinds={'Data': {ct.FlowKind.LazyValueFunc, ct.FlowKind.Info}},
	)
	def compute_data(self, props: dict, input_sockets: dict):
		lazy_value_func = input_sockets['Data'][ct.FlowKind.LazyValueFunc]
		info = input_sockets['Data'][ct.FlowKind.Info]

		# Check Flow
		if any(ct.FlowSignal.check(inp) for inp in [info, lazy_value_func]):
			return ct.FlowSignal.FlowPending

		# Compute Function Arguments
		operation = props['operation']
		if operation == 'NONE':
			return ct.FlowSignal.FlowPending

		## Dimension(s)
		dim_0 = props['dim_0']
		dim_1 = props['dim_1']
		if operation in ['PIN_LEN_ONE', 'PIN', 'SWAP'] and dim_0 == 'NONE':
			return ct.FlowSignal.FlowPending
		if operation == 'SWAP' and dim_1 == 'NONE':
			return ct.FlowSignal.FlowPending

		## Axis/Axes
		axis_0 = info.dim_names.index(dim_0) if dim_0 != 'NONE' else None
		axis_1 = info.dim_names.index(dim_1) if dim_1 != 'NONE' else None

		# Compose Output Function
		filter_func = {
			# Dimensions
			'PIN_LEN_ONE': lambda data: jnp.squeeze(data, axis_0),
			'PIN': lambda data, fixed_axis_idx: jnp.take(
				data, fixed_axis_idx, axis=axis_0
			),
			'SWAP': lambda data: jnp.swapaxes(data, axis_0, axis_1),
			# Interpret
			'DIM_TO_VEC': lambda data: data,
			'DIMS_TO_MAT': lambda data: data,
		}[props['operation']]

		return lazy_value_func.compose_within(
			filter_func,
			enclosing_func_args=[int] if operation == 'PIN' else [],
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

		# Check Flow
		if any(ct.FlowSignal.check(inp) for inp in [lazy_value_func, params]):
			return ct.FlowSignal.FlowPending

		return ct.ArrayFlow(
			values=lazy_value_func.func_jax(*params.func_args, **params.func_kwargs),
			unit=None,
		)

	####################
	# - Compute Auxiliary: Info
	####################
	@events.computes_output_socket(
		'Data',
		kind=ct.FlowKind.Info,
		props={'dim_0', 'dim_1', 'operation'},
		input_sockets={'Data'},
		input_socket_kinds={'Data': ct.FlowKind.Info},
	)
	def compute_data_info(self, props: dict, input_sockets: dict) -> ct.InfoFlow:
		info = input_sockets['Data']

		# Check Flow
		if ct.FlowSignal.check(info):
			return ct.FlowSignal.FlowPending

		# Collect Information
		dim_0 = props['dim_0']
		dim_1 = props['dim_1']

		if props['operation'] in ['PIN_LEN_ONE', 'PIN', 'SWAP'] and dim_0 == 'NONE':
			return ct.FlowSignal.FlowPending
		if props['operation'] == 'SWAP' and dim_1 == 'NONE':
			return ct.FlowSignal.FlowPending

		return {
			# Dimensions
			'PIN_LEN_ONE': lambda: info.delete_dimension(dim_0),
			'PIN': lambda: info.delete_dimension(dim_0),
			'SWAP': lambda: info.swap_dimensions(dim_0, dim_1),
			# Interpret
			'DIM_TO_VEC': lambda: info.shift_last_input,
			'DIMS_TO_MAT': lambda: info.shift_last_input.shift_last_input,
		}[props['operation']]()

	####################
	# - Compute Auxiliary: Info
	####################
	@events.computes_output_socket(
		'Data',
		kind=ct.FlowKind.Params,
		props={'dim_0', 'dim_1', 'operation'},
		input_sockets={'Data', 'Value'},
		input_socket_kinds={'Data': {ct.FlowKind.Info, ct.FlowKind.Params}},
		input_sockets_optional={'Value': True},
	)
	def compute_composed_params(
		self, props: dict, input_sockets: dict
	) -> ct.ParamsFlow:
		info = input_sockets['Data'][ct.FlowKind.Info]
		params = input_sockets['Data'][ct.FlowKind.Params]

		# Check Flow
		if any(ct.FlowSignal.check(inp) for inp in [info, params]):
			return ct.FlowSignal.FlowPending

		# Collect Information
		## Dimensions
		dim_0 = props['dim_0']
		dim_1 = props['dim_1']

		if props['operation'] in ['PIN_LEN_ONE', 'PIN', 'SWAP'] and dim_0 == 'NONE':
			return ct.FlowSignal.FlowPending
		if props['operation'] == 'SWAP' and dim_1 == 'NONE':
			return ct.FlowSignal.FlowPending

		## Pinned Value
		pinned_value = input_sockets['Value']
		has_pinned_value = not ct.FlowSignal.check(pinned_value)

		if props['operation'] == 'PIN' and has_pinned_value:
			# Compute IDX Corresponding to Dimension Index
			nearest_idx_to_value = info.dim_idx[dim_0].nearest_idx_of(
				input_sockets['Value'], require_sorted=True
			)

			return params.compose_within(enclosing_func_args=[nearest_idx_to_value])

		return params


####################
# - Blender Registration
####################
BL_REGISTER = [
	FilterMathNode,
]
BL_NODES = {ct.NodeType.FilterMath: (ct.NodeCategory.MAXWELLSIM_ANALYSIS_MATH)}
