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
	operation: bpy.props.EnumProperty(
		name='Op',
		description='Operation to filter with',
		items=lambda self, _: self.search_operations(),
		update=lambda self, context: self.on_prop_changed('operation', context),
	)

	dim: bpy.props.StringProperty(
		name='Dim',
		description='Dims to use when filtering data',
		default='',
		search=lambda self, _, edit_text: self.search_dims(edit_text),
		update=lambda self, context: self.on_prop_changed('dim', context),
	)

	dim_names: list[str] = bl_cache.BLField([])
	dim_lens: dict[str, int] = bl_cache.BLField({})

	@property
	def has_dim(self) -> bool:
		return (
			self.active_socket_set in ['By Dim', 'By Dim Value']
			and self.inputs['Data'].is_linked
			and self.dim_names
		)

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

		return items

	####################
	# - Dim Search
	####################
	def search_dims(self, edit_text: str) -> list[tuple[str, str, str]]:
		if self.has_dim:
			dims = [
				(dim_name, dim_name)
				for dim_name in self.dim_names
				if edit_text == '' or edit_text.lower() in dim_name.lower()
			]

			# Squeeze: Dimension Must Have Length=1
			if self.operation == 'SQUEEZE':
				return [dim for dim in dims if self.dim_lens[dim[0]] == 1]
			return dims
		return []

	####################
	# - UI
	####################
	def draw_props(self, _: bpy.types.Context, layout: bpy.types.UILayout) -> None:
		layout.prop(self, 'operation', text='')
		if self.has_dim:
			layout.prop(self, 'dim', text='')

	####################
	# - Events
	####################
	@events.on_value_changed(
		socket_name={'Data'},
		prop_name={'active_socket_set', 'dim'},
		props={'active_socket_set', 'dim'},
		input_sockets={'Data'},
		input_socket_kinds={'Data': ct.FlowKind.Info},
		input_sockets_optional={'Data': True},
	)
	def on_any_change(self, props: dict, input_sockets: dict):
		# Set Dimension Names from InfoFlow
		if input_sockets['Data'].dim_names:
			self.dim_names = input_sockets['Data'].dim_names
			self.dim_lens = {
				dim_name: len(dim_idx)
				for dim_name, dim_idx in input_sockets['Data'].dim_idx.items()
			}
		else:
			self.dim_names = []
			self.dim_lens = {}

		# Add Input Value w/Unit from InfoFlow
		## Socket Type is determined from the Unit
		if (
			props['active_socket_set'] == 'By Dim Value'
			and props['dim'] != ''
			and props['dim'] in input_sockets['Data'].dim_names
		):
			socket_def = sockets.SOCKET_DEFS[
				ct.unit_to_socket_type(input_sockets['Data'].dim_idx[props['dim']].unit)
			]
			if (
				_val_socket_def := self.loose_input_sockets.get('Value')
			) is None or _val_socket_def != socket_def:
				self.loose_input_sockets = {
					'Value': socket_def(),
				}
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
		lazy_value_func = input_sockets['Data'][ct.FlowKind.LazyValueFunc]
		info = input_sockets['Data'][ct.FlowKind.Info]

		# Determine Bound/Free Parameters
		if props['dim'] in info.dim_names:
			axis = info.dim_names.index(props['dim'])
		else:
			msg = 'Dimension invalid'
			raise ValueError(msg)

		func_args = [int] if props['active_socket_set'] == 'By Dim Value' else []

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
		lazy_value_func = output_sockets['Data'][ct.FlowKind.LazyValueFunc]
		params = output_sockets['Data'][ct.FlowKind.Params]
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
		info = input_sockets['Data']

		if props['dim'] in info.dim_names:
			axis = info.dim_names.index(props['dim'])
		else:
			return ct.InfoFlow()

		# Compute Axis
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
			)

		return ct.InfoFlow()

	@events.computes_output_socket(
		'Data',
		kind=ct.FlowKind.Params,
		props={'active_socket_set', 'dim', 'operation'},
		input_sockets={'Data', 'Value'},
		input_socket_kinds={'Data': {ct.FlowKind.Info, ct.FlowKind.Params}},
		input_sockets_optional={'Value': True},
	)
	def compute_data_params(self, props: dict, input_sockets: dict) -> ct.ParamsFlow:
		info = input_sockets['Data'][ct.FlowKind.Info]
		params = input_sockets['Data'][ct.FlowKind.Params]

		if (
			(props['active_socket_set'], props['operation'])
			in [
				('By Dim Value', 'FIX'),
			]
			and props['dim'] in info.dim_names
			and input_sockets['Value'] is not None
		):
			# Compute IDX Corresponding to Value
			## Aka. "indexing by a float"
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


## TODO TODO Okay so just like, Value needs to be a Loose socket, events needs to be able to handle sets of kinds, the invalidator needs to be able to handle sets of kinds too. Given all that, we only need to propagate the output array unit; given all all that, we are 100% goddamn ready to fix that goddamn coordinate.
