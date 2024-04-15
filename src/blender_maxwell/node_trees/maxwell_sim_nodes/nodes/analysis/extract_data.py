import typing as typ

import bpy

from .....utils import logger
from ... import contracts as ct
from ... import sockets
from .. import base, events

log = logger.get(__name__)

CACHE_SIM_DATA = {}


class ExtractDataNode(base.MaxwellSimNode):
	"""Node for extracting data from other objects."""

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
	# - Properties: Sim Data
	####################
	sim_data__monitor_name: bpy.props.EnumProperty(
		name='Sim Data Monitor Name',
		description='Monitor to extract from the attached SimData',
		items=lambda self, context: self.search_monitors(context),
		update=lambda self, context: self.sync_prop('sim_data__monitor_name', context),
	)

	cache__num_monitors: bpy.props.StringProperty(default='')
	cache__monitor_names: bpy.props.StringProperty(default='')
	cache__monitor_types: bpy.props.StringProperty(default='')

	def search_monitors(self, _: bpy.types.Context) -> list[tuple[str, str, str]]:
		"""Search the linked simulation data for monitors."""
		# No Linked Sim Data: Return 'None'
		if not self.inputs.get('Sim Data') or not self.inputs['Sim Data'].is_linked:
			return [('NONE', 'None', 'No monitors')]

		# Return Monitor Names
		## Special Case for No Monitors
		monitor_names = (
			self.cache__monitor_names.split(',') if self.cache__monitor_names else []
		)
		monitor_types = (
			self.cache__monitor_types.split(',') if self.cache__monitor_types else []
		)
		if len(monitor_names) == 0:
			return [('NONE', 'None', 'No monitors')]
		return [
			(
				monitor_name,
				f'{monitor_name}',
				f'Monitor "{monitor_name}" ({monitor_type}) recorded by the Sim',
			)
			for monitor_name, monitor_type in zip(
				monitor_names, monitor_types, strict=False
			)
		]

	def draw_props__sim_data(
		self, _: bpy.types.Context, col: bpy.types.UILayout
	) -> None:
		col.prop(self, 'sim_data__monitor_name', text='')

	def draw_info__sim_data(
		self, _: bpy.types.Context, col: bpy.types.UILayout
	) -> None:
		if self.sim_data__monitor_name != 'NONE':
			# Header
			row = col.row()
			row.alignment = 'CENTER'
			row.label(text=f'{self.cache__num_monitors} Monitors')

			# Monitor Info
			if int(self.cache__num_monitors) > 0:
				for monitor_name, monitor_type in zip(
					self.cache__monitor_names.split(','),
					self.cache__monitor_types.split(','),
					strict=False,
				):
					col.label(text=f'{monitor_name}: {monitor_type}')

	####################
	# - Events: Sim Data
	####################
	@events.on_value_changed(
		socket_name='Sim Data',
	)
	def on_sim_data_changed(self):
		# SimData Cache Hit and SimData Input Unlinked
		## Delete Cache Entry
		if (
			CACHE_SIM_DATA.get(self.instance_id) is not None
			and not self.inputs['Sim Data'].is_linked
		):
			CACHE_SIM_DATA.pop(self.instance_id, None)  ## Both member-check
			self.cache__num_monitors = ''
			self.cache__monitor_names = ''
			self.cache__monitor_types = ''

		# SimData Cache Miss and Linked SimData
		if (
			CACHE_SIM_DATA.get(self.instance_id) is None
			and self.inputs['Sim Data'].is_linked
		):
			sim_data = self._compute_input('Sim Data')

			## Create Cache Entry
			CACHE_SIM_DATA[self.instance_id] = {
				'sim_data': sim_data,
				'monitor_names': list(sim_data.monitor_data.keys()),
				'monitor_types': [
					monitor_data.type for monitor_data in sim_data.monitor_data.values()
				],
			}
			cache = CACHE_SIM_DATA[self.instance_id]
			self.cache__num_monitors = str(len(cache['monitor_names']))
			self.cache__monitor_names = ','.join(cache['monitor_names'])
			self.cache__monitor_types = ','.join(cache['monitor_types'])

	####################
	# - Properties: Field Data
	####################
	field_data__component: bpy.props.EnumProperty(
		name='Field Data Component',
		description='Field monitor component to extract from the attached Field Data',
		items=lambda self, context: self.search_field_data_components(context),
		update=lambda self, context: self.sync_prop('field_data__component', context),
	)

	cache__components: bpy.props.StringProperty(default='')

	def search_field_data_components(
		self, _: bpy.types.Context
	) -> list[tuple[str, str, str]]:
		if not self.inputs.get('Field Data') or not self.inputs['Field Data'].is_linked:
			return [('NONE', 'None', 'No data')]

		if not self.cache__components:
			return [('NONE', 'Loading...', 'Loading data...')]

		components = [
			tuple(component_str.split(','))
			for component_str in self.cache__components.split('|')
		]

		if len(components) == 0:
			return [('NONE', 'None', 'No components')]
		return components

	def draw_props__field_data(
		self, _: bpy.types.Context, col: bpy.types.UILayout
	) -> None:
		col.prop(self, 'field_data__component', text='')

	def draw_info__field_data(
		self, _: bpy.types.Context, col: bpy.types.UILayout
	) -> None:
		pass

	####################
	# - Events: Field Data
	####################
	@events.on_value_changed(
		socket_name='Field Data',
	)
	def on_field_data_changed(self):
		if self.inputs['Field Data'].is_linked and not self.cache__components:
			field_data = self._compute_input('Field Data')
			components = [
				*([('Ex', 'Ex', 'Ex')] if field_data.Ex is not None else []),
				*([('Ey', 'Ey', 'Ey')] if field_data.Ey is not None else []),
				*([('Ez', 'Ez', 'Ez')] if field_data.Ez is not None else []),
				*([('Hx', 'Hx', 'Hx')] if field_data.Hx is not None else []),
				*([('Hy', 'Hy', 'Hy')] if field_data.Hy is not None else []),
				*([('Hz', 'Hz', 'Hz')] if field_data.Hz is not None else []),
			]
			self.cache__components = '|'.join(
				[','.join(component) for component in components]
			)

		elif not self.inputs['Field Data'].is_linked and self.cache__components:
			self.cache__components = ''

	####################
	# - Flux Data
	####################

	def draw_props__flux_data(
		self, _: bpy.types.Context, col: bpy.types.UILayout
	) -> None:
		pass

	def draw_info__flux_data(
		self, _: bpy.types.Context, col: bpy.types.UILayout
	) -> None:
		pass

	####################
	# - Global
	####################
	def draw_props(self, context: bpy.types.Context, col: bpy.types.UILayout) -> None:
		if self.active_socket_set == 'Sim Data':
			self.draw_props__sim_data(context, col)
		if self.active_socket_set == 'Field Data':
			self.draw_props__field_data(context, col)
		if self.active_socket_set == 'Flux Data':
			self.draw_props__flux_data(context, col)

	def draw_info(self, context: bpy.types.Context, col: bpy.types.UILayout) -> None:
		if self.active_socket_set == 'Sim Data':
			self.draw_info__sim_data(context, col)
		if self.active_socket_set == 'Field Data':
			self.draw_info__field_data(context, col)
		if self.active_socket_set == 'Flux Data':
			self.draw_info__flux_data(context, col)

	@events.computes_output_socket(
		'Data',
		props={'sim_data__monitor_name', 'field_data__component'},
	)
	def compute_extracted_data(self, props: dict):
		if self.active_socket_set == 'Sim Data':
			if (
				CACHE_SIM_DATA.get(self.instance_id) is None
				and self.inputs['Sim Data'].is_linked
			):
				self.on_sim_data_changed()

			sim_data = CACHE_SIM_DATA[self.instance_id]['sim_data']
			return sim_data.monitor_data[props['sim_data__monitor_name']]

		elif self.active_socket_set == 'Field Data':  # noqa: RET505
			field_data = self._compute_input('Field Data')
			return getattr(field_data, props['field_data__component'])

		elif self.active_socket_set == 'Flux Data':
			flux_data = self._compute_input('Flux Data')
			return flux_data.flux

		msg = f'Tried to get data from unknown output socket in "{self.bl_label}"'
		raise RuntimeError(msg)


####################
# - Blender Registration
####################
BL_REGISTER = [
	ExtractDataNode,
]
BL_NODES = {ct.NodeType.ExtractData: (ct.NodeCategory.MAXWELLSIM_ANALYSIS)}
