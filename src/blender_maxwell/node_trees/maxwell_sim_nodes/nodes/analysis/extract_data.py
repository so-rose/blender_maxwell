import typing as typ

import bpy
import jax.numpy as jnp
import sympy.physics.units as spu

from blender_maxwell.utils import bl_cache, logger

from ... import contracts as ct
from ... import sockets
from .. import base, events

log = logger.get(__name__)


class ExtractDataNode(base.MaxwellSimNode):
	"""Node for extracting data from particular objects."""

	node_type = ct.NodeType.ExtractData
	bl_label = 'Extract'

	input_socket_sets: typ.ClassVar = {
		'Sim Data': {'Sim Data': sockets.MaxwellFDTDSimDataSocketDef()},
		'Field Data': {'Field Data': sockets.AnySocketDef()},
		'Flux Data': {'Flux Data': sockets.AnySocketDef()},
	}
	output_sockets: typ.ClassVar = {
		'Data': sockets.AnySocketDef(),
	}

	####################
	# - Properties
	####################
	extract_filter: bpy.props.EnumProperty(
		name='Extract Filter',
		description='Data to extract from the input',
		search=lambda self, _, edit_text: self.search_extract_filters(edit_text),
		update=lambda self, context: self.on_prop_changed('extract_filter', context),
	)

	# Sim Data
	sim_data_monitor_nametype: dict[str, str] = bl_cache.BLField({})

	# Field Data
	field_data_components: set[str] = bl_cache.BLField(set())

	def search_extract_filters(
		self, _: bpy.types.Context
	) -> list[tuple[str, str, str]]:
		# Sim Data
		if self.active_socket_set == 'Sim Data' and self.inputs['Sim Data'].is_linked:
			return [
				(
					monitor_name,
					f'{monitor_name}',
					f'Monitor "{monitor_name}" ({monitor_type}) recorded by the Sim',
				)
				for monitor_name, monitor_type in self.sim_data_monitor_nametype.items()
			]

		# Field Data
		if self.active_socket_set == 'Field Data' and self.inputs['Sim Data'].is_linked:
			return [
				([('Ex', 'Ex', 'Ex')] if 'Ex' in self.field_data_components else [])
				+ ([('Ey', 'Ey', 'Ey')] if 'Ey' in self.field_data_components else [])
				+ ([('Ez', 'Ez', 'Ez')] if 'Ez' in self.field_data_components else [])
				+ ([('Hx', 'Hx', 'Hx')] if 'Hx' in self.field_data_components else [])
				+ ([('Hy', 'Hy', 'Hy')] if 'Hy' in self.field_data_components else [])
				+ ([('Hz', 'Hz', 'Hz')] if 'Hz' in self.field_data_components else [])
			]

		# Flux Data
		## Nothing to extract.

		# Fallback
		return []

	####################
	# - UI
	####################
	def draw_props(self, _: bpy.types.Context, col: bpy.types.UILayout) -> None:
		col.prop(self, 'extract_filter', text='')

	def draw_info(self, _: bpy.types.Context, col: bpy.types.UILayout) -> None:
		if self.active_socket_set == 'Sim Data' and self.inputs['Sim Data'].is_linked:
			# Header
			row = col.row()
			row.alignment = 'CENTER'
			row.label(text=f'{self.cache__num_monitors} Monitors')

			# Monitor Info
			if len(self.sim_data_monitor_nametype) > 0:
				for (
					monitor_name,
					monitor_type,
				) in self.sim_data_monitor_nametype.items():
					col.label(text=f'{monitor_name}: {monitor_type}')

	####################
	# - Events
	####################
	@events.on_value_changed(
		socket_name='Sim Data',
		input_sockets={'Sim Data'},
		input_sockets_optional={'Sim Data': True},
	)
	def on_sim_data_changed(self, input_sockets: dict):
		if input_sockets['Sim Data'] is not None:
			self.sim_data_monitor_nametype = {
				monitor_name: monitor_data.type
				for monitor_name, monitor_data in input_sockets[
					'Sim Data'
				].monitor_data.items()
			}

	@events.on_value_changed(
		socket_name='Field Data',
		input_sockets={'Field Data'},
		input_sockets_optional={'Field Data': True},
	)
	def on_field_data_changed(self, input_sockets: dict):
		if input_sockets['Field Data'] is not None:
			self.field_data_components = (
				{'Ex'}
				if input_sockets['Field Data'].Ex is not None
				else set() | {'Ey'}
				if input_sockets['Field Data'].Ey is not None
				else set() | {'Ez'}
				if input_sockets['Field Data'].Ez is not None
				else set() | {'Hx'}
				if input_sockets['Field Data'].Hx is not None
				else set() | {'Hy'}
				if input_sockets['Field Data'].Hy is not None
				else set() | {'Hz'}
				if input_sockets['Field Data'].Hz is not None
				else set()
			)

	####################
	# - Output: Value
	####################
	@events.computes_output_socket(
		'Data',
		kind=ct.FlowKind.Value,
		props={'active_socket_set', 'extract_filter'},
		input_sockets={'Sim Data', 'Field Data', 'Flux Data'},
		input_sockets_optional={
			'Sim Data': True,
			'Field Data': True,
			'Flux Data': True,
		},
	)
	def compute_extracted_data(self, props: dict, input_sockets: dict):
		if props['active_socket_set'] == 'Sim Data':
			return input_sockets['Sim Data'].monitor_data[props['extract_filter']]

		if props['active_socket_set'] == 'Field Data':
			return getattr(input_sockets['Field Data'], props['extract_filter'])

		if props['active_socket_set'] == 'Flux Data':
			return input_sockets['Flux Data']

		msg = f'Tried to get a "FlowKind.Value" from socket set {props["active_socket_set"]} in "{self.bl_label}"'
		raise RuntimeError(msg)

	####################
	# - Output: LazyValueFunc
	####################
	@events.computes_output_socket(
		'Data',
		kind=ct.FlowKind.LazyValueFunc,
		props={'active_socket_set'},
		output_sockets={'Data'},
		output_socket_kinds={'Data': ct.FlowKind.Value},
	)
	def compute_extracted_data_lazy(self, props: dict, output_sockets: dict):
		if self.active_socket_set in {'Field Data', 'Flux Data'}:
			data = jnp.array(output_sockets['Data'].data)
			return ct.LazyValueFuncFlow(func=lambda: data, supports_jax=True)

		msg = f'Tried to get a "FlowKind.LazyValueFunc" from socket set {props["active_socket_set"]} in "{self.bl_label}"'
		raise RuntimeError(msg)

	####################
	# - Output: Info
	####################
	@events.computes_output_socket(
		'Data',
		kind=ct.FlowKind.Info,
		props={'active_socket_set'},
		output_sockets={'Data'},
		output_socket_kinds={'Data': ct.FlowKind.Value},
	)
	def compute_extracted_data_info(self, props: dict, output_sockets: dict):
		if props['active_socket_set'] == 'Field Data':
			xarr = output_sockets['Data']
			return ct.InfoFlow(
				dim_names=['x', 'y', 'z', 'f'],
				dim_idx={
					axis: ct.ArrayFlow(values=xarr.get_index(axis).values, unit=spu.um)
					for axis in ['x', 'y', 'z']
				}
				| {
					'f': ct.ArrayFlow(
						values=xarr.get_index('f').values, unit=spu.hertz
					),
				},
			)

		if props['active_socket_set'] == 'Flux Data':
			xarr = output_sockets['Data']
			return ct.InfoFlow(
				dim_names=['f'],
				dim_idx={
					'f': ct.ArrayFlow(
						values=xarr.get_index('f').values, unit=spu.hertz
					),
				},
			)

		msg = f'Tried to get a "FlowKind.Info" from socket set {props["active_socket_set"]} in "{self.bl_label}"'
		raise RuntimeError(msg)


####################
# - Blender Registration
####################
BL_REGISTER = [
	ExtractDataNode,
]
BL_NODES = {ct.NodeType.ExtractData: (ct.NodeCategory.MAXWELLSIM_ANALYSIS)}
