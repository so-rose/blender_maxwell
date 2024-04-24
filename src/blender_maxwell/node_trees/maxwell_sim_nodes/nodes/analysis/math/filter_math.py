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
	"""Reduces the dimensionality of data.

	Attributes:
		operation: Operation to apply to the input.
		dim: Dims to use when filtering data
	"""

	node_type = ct.NodeType.FilterMath
	bl_label = 'Filter Math'

	input_sockets: typ.ClassVar = {
		'Data': sockets.DataSocketDef(format='jax'),
	}
	input_socket_sets: typ.ClassVar = {
		'By Dim': {},
		'By Dim Value': {},
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

	dim: enum.Enum = bl_cache.BLField(
		None, prop_ui=True, enum_cb=lambda self, _: self.search_dims()
	)

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
		if self.active_socket_set == 'By Dim':
			items += [
				('SQUEEZE', 'del a | #=1', 'Squeeze'),
			]
		if self.active_socket_set == 'By Dim Value':
			items += [
				('FIX', 'del a | i≈v', 'Fix Coordinate'),
			]

		return [(*item, '', i) for i, item in enumerate(items)]

	####################
	# - Dim Search
	####################
	def search_dims(self) -> list[ct.BLEnumElement]:
		if self.data_info is not None:
			dims = [
				(dim_name, dim_name, dim_name, '', i)
				for i, dim_name in enumerate(self.data_info.dim_names)
			]

			# Squeeze: Dimension Must Have Length=1
			## We must also correct the "NUMBER" of the enum.
			if self.operation == 'SQUEEZE':
				filtered_dims = [
					dim for dim in dims if self.data_info.dim_lens[dim[0]] == 1
				]
				return [(*dim[:-1], i) for i, dim in enumerate(filtered_dims)]

			return dims
		return []

	####################
	# - UI
	####################
	def draw_props(self, _: bpy.types.Context, layout: bpy.types.UILayout) -> None:
		layout.prop(self, self.blfields['operation'], text='')
		if self.data_info is not None and self.data_info.dim_names:
			layout.prop(self, self.blfields['dim'], text='')

	####################
	# - Events
	####################
	@events.on_value_changed(
		socket_name='Data',
		prop_name='active_socket_set',
		run_on_init=True,
		input_sockets={'Data'},
	)
	def on_any_change(self, input_sockets: dict):
		if all(
			not ct.FlowSignal.check_single(
				input_socket_value, ct.FlowSignal.FlowPending
			)
			for input_socket_value in input_sockets.values()
		):
			self.operation = bl_cache.Signal.ResetEnumItems
			self.dim = bl_cache.Signal.ResetEnumItems

	@events.on_value_changed(
		socket_name='Data',
		prop_name='dim',
		## run_on_init: Implicitly triggered.
		props={'active_socket_set', 'dim'},
		input_sockets={'Data'},
		input_socket_kinds={'Data': ct.FlowKind.Info},
	)
	def on_dim_change(self, props: dict, input_sockets: dict):
		if input_sockets['Data'] == ct.FlowSignal.FlowPending:
			return

		# Add/Remove Input Socket "Value"
		if (
			not ct.Flowsignal.check(input_sockets['Data'])
			and props['active_socket_set'] == 'By Dim Value'
			and props['dim'] != 'NONE'
		):
			# Get Current and Wanted Socket Defs
			current_bl_socket = self.loose_input_sockets.get('Value')
			wanted_socket_def = sockets.SOCKET_DEFS[
				ct.unit_to_socket_type(input_sockets['Data'].dim_idx[props['dim']].unit)
			]

			# Determine Whether to Declare New Loose Input SOcket
			if (
				current_bl_socket is None
				or sockets.SOCKET_DEFS[current_bl_socket.socket_type]
				!= wanted_socket_def
			):
				self.loose_input_sockets = {
					'Value': wanted_socket_def(),
				}  ## TODO: Can we do the boilerplate in base.py?
		elif self.loose_input_sockets:
			self.loose_input_sockets = {}

	####################
	# - Compute: LazyValueFunc / Array
	####################
	@events.computes_output_socket(
		'Data',
		kind=ct.FlowKind.LazyValueFunc,
		props={'active_socket_set', 'operation', 'dim'},
		input_sockets={'Data'},
		input_socket_kinds={'Data': {ct.FlowKind.LazyValueFunc, ct.FlowKind.Info}},
	)
	def compute_data(self, props: dict, input_sockets: dict):
		# Retrieve Inputs
		lazy_value_func = input_sockets['Data'][ct.FlowKind.LazyValueFunc]
		info = input_sockets['Data'][ct.FlowKind.Info]

		# Check Flow
		if (
			any(ct.FlowSignal.check(inp) for inp in [info, lazy_value_func])
			or props['operation'] == 'NONE'
		):
			return ct.FlowSignal.FlowPending

		# Compute Bound/Free Parameters
		func_args = [int] if props['active_socket_set'] == 'By Dim Value' else []
		axis = info.dim_names.index(props['dim'])

		# Select Function
		filter_func: typ.Callable[[jax.Array], jax.Array] = {
			'By Dim': {'SQUEEZE': lambda data: jnp.squeeze(data, axis)},
			'By Dim Value': {
				'FIX': lambda data, fixed_axis_idx: jnp.take(
					data, fixed_axis_idx, axis=axis
				)
			},
		}[props['active_socket_set']][props['operation']]

		# Compose Function for Output
		return lazy_value_func.compose_within(
			filter_func,
			enclosing_func_args=func_args,
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
		# Retrieve Inputs
		lazy_value_func = output_sockets['Data'][ct.FlowKind.LazyValueFunc]
		params = output_sockets['Data'][ct.FlowKind.Params]

		# Check Flow
		if any(ct.FlowSignal.check(inp) for inp in [lazy_value_func, params]):
			return ct.FlowSignal.FlowPending

		# Compute Array
		return ct.ArrayFlow(
			values=lazy_value_func.func_jax(*params.func_args, **params.func_kwargs),
			unit=None,  ## TODO: Unit Propagation
		)

	####################
	# - Compute Auxiliary: Info / Params
	####################
	@events.computes_output_socket(
		'Data',
		kind=ct.FlowKind.Info,
		props={'active_socket_set', 'dim', 'operation'},
		input_sockets={'Data'},
		input_socket_kinds={'Data': ct.FlowKind.Info},
	)
	def compute_data_info(self, props: dict, input_sockets: dict) -> ct.InfoFlow:
		# Retrieve Inputs
		info = input_sockets['Data']

		# Check Flow
		if ct.FlowSignal.check(info) or props['dim'] == 'NONE':
			return ct.FlowSignal.FlowPending

		# Compute Information
		## Compute Info w/By-Operation Change to Dimensions
		axis = info.dim_names.index(props['dim'])

		if (props['active_socket_set'], props['operation']) in [
			('By Dim', 'SQUEEZE'),
			('By Dim Value', 'FIX'),
		] and info.dim_names:
			return ct.InfoFlow(
				dim_names=info.dim_names[:axis] + info.dim_names[axis + 1 :],
				dim_idx={
					dim_name: dim_idx
					for dim_name, dim_idx in info.dim_idx.items()
					if dim_name != props['dim']
				},
				output_names=info.output_names,
				output_mathtypes=info.output_mathtypes,
				output_units=info.output_units,
			)

		msg = f'Active socket set {props["active_socket_set"]} and operation {props["operation"]} don\'t have an InfoFlow defined'
		raise RuntimeError(msg)

	@events.computes_output_socket(
		'Data',
		kind=ct.FlowKind.Params,
		props={'active_socket_set', 'dim', 'operation'},
		input_sockets={'Data', 'Value'},
		input_socket_kinds={'Data': {ct.FlowKind.Info, ct.FlowKind.Params}},
		input_sockets_optional={'Value': True},
	)
	def compute_composed_params(
		self, props: dict, input_sockets: dict
	) -> ct.ParamsFlow:
		# Retrieve Inputs
		info = input_sockets['Data'][ct.FlowKind.Info]
		params = input_sockets['Data'][ct.FlowKind.Params]

		if any(ct.FlowSignal.check(inp) for inp in [info, params]):
			return ct.FlowSignal.FlowPending

		# Compute Composed Parameters
		## -> Only operations that add parameters.
		## -> A dimension must be selected.
		## -> There must be an input value.
		if (
			(props['active_socket_set'], props['operation'])
			in [
				('By Dim Value', 'FIX'),
			]
			and props['dim'] != 'NONE'
			and not ct.FlowSignal.check(input_sockets['Value'])
		):
			# Compute IDX Corresponding to Coordinate Value
			## -> Each dimension declares a unit-aware real number at each index.
			## -> "Value" is a unit-aware real number from loose input socket.
			## -> This finds the dimensional index closest to "Value".
			## Total Effect: Indexing by a unit-aware real number.
			nearest_idx_to_value = info.dim_idx[props['dim']].nearest_idx_of(
				input_sockets['Value'], require_sorted=True
			)

			# Compose Parameters
			return params.compose_within(enclosing_func_args=[nearest_idx_to_value])

		return params


####################
# - Blender Registration
####################
BL_REGISTER = [
	FilterMathNode,
]
BL_NODES = {ct.NodeType.FilterMath: (ct.NodeCategory.MAXWELLSIM_ANALYSIS_MATH)}
