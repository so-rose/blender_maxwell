import typing as typ

import bpy
import jax
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
		'Monitor Data': {'Monitor Data': sockets.DataSocketDef(format='monitor_data')},
	}
	output_socket_sets: typ.ClassVar = {
		'Sim Data': {'Monitor Data': sockets.DataSocketDef(format='monitor_data')},
		'Monitor Data': {'Data': sockets.DataSocketDef(format='jax')},
	}

	####################
	# - Properties
	####################
	extract_filter: bpy.props.StringProperty(
		name='Extract Filter',
		description='Data to extract from the input',
		search=lambda self, _, edit_text: self.search_extract_filters(edit_text),
		update=lambda self, context: self.on_prop_changed('extract_filter', context),
	)

	# Sim Data
	sim_data_monitor_nametype: dict[str, str] = bl_cache.BLField({})

	# Monitor Data
	monitor_data_type: str = bl_cache.BLField('')
	monitor_data_components: list[str] = bl_cache.BLField([])

	####################
	# - Computed Properties
	####################
	@bl_cache.cached_bl_property(persist=False)
	def has_sim_data(self) -> bool:
		return (
			self.active_socket_set == 'Sim Data'
			and self.inputs['Sim Data'].is_linked
			and self.sim_data_monitor_nametype
		)

	@bl_cache.cached_bl_property(persist=False)
	def has_monitor_data(self) -> bool:
		return (
			self.active_socket_set == 'Monitor Data'
			and self.inputs['Monitor Data'].is_linked
			and self.monitor_data_type
		)

	####################
	# - Extraction Filter Search
	####################
	def search_extract_filters(self, edit_text: str) -> list[tuple[str, str, str]]:
		if self.has_sim_data:
			return [
				(
					monitor_name,
					monitor_type.removesuffix('Data'),
				)
				for monitor_name, monitor_type in self.sim_data_monitor_nametype.items()
				if edit_text == '' or edit_text.lower() in monitor_name.lower()
			]

		if self.has_monitor_data:
			return [
				(component_name, f'ℂ {component_name[1]}-Pol')
				for component_name in self.monitor_data_components
				if (edit_text == '' or edit_text.lower() in component_name.lower())
			]

		return []

	####################
	# - UI
	####################
	def draw_props(self, _: bpy.types.Context, col: bpy.types.UILayout) -> None:
		col.prop(self, 'extract_filter', text='')

	def draw_info(self, _: bpy.types.Context, col: bpy.types.UILayout) -> None:
		if self.has_sim_data or self.has_monitor_data:
			# Header
			row = col.row()
			row.alignment = 'CENTER'
			if self.has_sim_data:
				row.label(text=f'{len(self.sim_data_monitor_nametype)} Monitors')
			elif self.has_monitor_data:
				row.label(text=f'{self.monitor_data_type} Monitor Data')

			# Monitor Data Contents
			row = col.row()
			box = row.box()
			grid = box.grid_flow(row_major=True, columns=2, even_columns=True)
			for name, desc in self.search_extract_filters(edit_text=''):
				grid.label(text=name)
				grid.label(text=desc if desc else '')

	####################
	# - Events
	####################
	@events.on_value_changed(
		socket_name={'Sim Data', 'Monitor Data'},
		prop_name='active_socket_set',
		input_sockets={'Sim Data', 'Monitor Data'},
		input_sockets_optional={'Sim Data': True, 'Monitor Data': True},
	)
	def on_sim_data_changed(self, input_sockets: dict):
		if input_sockets['Sim Data'] is not None:
			# Sim Data Monitors: Set Name -> Type
			self.sim_data_monitor_nametype = {
				monitor_name: monitor_data.type
				for monitor_name, monitor_data in input_sockets[
					'Sim Data'
				].monitor_data.items()
			}

		if input_sockets['Monitor Data'] is not None:
			# Monitor Data Type
			self.monitor_data_type = input_sockets['Monitor Data'].type.removesuffix(
				'Data'
			)

			# Field/FieldTime
			if self.monitor_data_type in ['Field', 'FieldTime']:
				self.monitor_data_components = [
					field_component
					for field_component in ['Ex', 'Ey', 'Ez', 'Hx', 'Hy', 'Hz']
					if hasattr(input_sockets['Monitor Data'], field_component)
				]

			# Permittivity
			if self.monitor_data_type == 'Permittivity':
				self.monitor_data_components = ['xx', 'yy', 'zz']

			# Flux/FluxTime
			if self.monitor_data_type in ['Flux', 'FluxTime']:
				self.monitor_data_components = ['flux']

			# FieldProjection(Angle/Cartesian/KSpace)/Diffraction
			if self.monitor_data_type in [
				'FieldProjectionAngle',
				'FieldProjectionCartesian',
				'FieldProjectionKSpace',
				'Diffraction',
			]:
				self.monitor_data_components = [
					'Er',
					'Etheta',
					'Ephi',
					'Hr',
					'Htheta',
					'Hphi',
				]

		# Invalidate Computed Property Caches
		self.has_sim_data = bl_cache.Signal.InvalidateCache
		self.has_monitor_data = bl_cache.Signal.InvalidateCache

		# Reset Extraction Filter
		## The extraction filter that was set before may not be valid anymore.
		## If so, simply remove it.
		if self.extract_filter not in [
			el[0] for el in self.search_extract_filters(edit_text='')
		]:
			self.extract_filter = ''

	####################
	# - Output: Value
	####################
	@events.computes_output_socket(
		'Monitor Data',
		kind=ct.FlowKind.Value,
		props={'extract_filter'},
		input_sockets={'Sim Data'},
	)
	def compute_monitor_data(self, props: dict, input_sockets: dict):
		return input_sockets['Sim Data'].monitor_data[props['extract_filter']]

	@events.computes_output_socket(
		'Data',
		kind=ct.FlowKind.Value,
		props={'extract_filter'},
		input_sockets={'Monitor Data'},
	)
	def compute_data(self, props: dict, input_sockets: dict) -> jax.Array:
		xarray_data = getattr(input_sockets['Monitor Data'], props['extract_filter'])
		return jnp.array(xarray_data.data)  ## TODO: Can it be done without a copy?

	####################
	# - Output: LazyValueFunc
	####################
	@events.computes_output_socket(
		'Data',
		kind=ct.FlowKind.LazyValueFunc,
		output_sockets={'Data'},
		output_socket_kinds={'Data': ct.FlowKind.Value},
	)
	def compute_extracted_data_lazy(self, output_sockets: dict) -> ct.LazyValueFuncFlow:
		return ct.LazyValueFuncFlow(
			func=lambda: output_sockets['Data'], supports_jax=True
		)

	####################
	# - Output: Info
	####################
	@events.computes_output_socket(
		'Data',
		kind=ct.FlowKind.Info,
		props={'monitor_data_type', 'extract_filter'},
		input_sockets={'Monitor Data'},
	)
	def compute_extracted_data_info(
		self, props: dict, input_sockets: dict
	) -> ct.InfoFlow:  # noqa: PLR0911
		if input_sockets['Monitor Data'] is None or not props['extract_filter']:
			return ct.InfoFlow()

		xarr = getattr(input_sockets['Monitor Data'], props['extract_filter'])

		# XYZF: Field / Permittivity / FieldProjectionCartesian
		if props['monitor_data_type'] in {
			'Field',
			'Permittivity',
			'FieldProjectionCartesian',
		}:
			return ct.InfoFlow(
				dim_names=['x', 'y', 'z', 'f'],
				dim_idx={
					axis: ct.ArrayFlow(
						values=xarr.get_index(axis).values, unit=spu.um, is_sorted=True
					)
					for axis in ['x', 'y', 'z']
				}
				| {
					'f': ct.ArrayFlow(
						values=xarr.get_index('f').values,
						unit=spu.hertz,
						is_sorted=True,
					),
				},
			)

		# XYZT: FieldTime
		if props['monitor_data_type'] == 'FieldTime':
			return ct.InfoFlow(
				dim_names=['x', 'y', 'z', 't'],
				dim_idx={
					axis: ct.ArrayFlow(
						values=xarr.get_index(axis).values, unit=spu.um, is_sorted=True
					)
					for axis in ['x', 'y', 'z']
				}
				| {
					't': ct.ArrayFlow(
						values=xarr.get_index('t').values,
						unit=spu.second,
						is_sorted=True,
					),
				},
			)

		# F: Flux
		if props['monitor_data_type'] == 'Flux':
			return ct.InfoFlow(
				dim_names=['f'],
				dim_idx={
					'f': ct.ArrayFlow(
						values=xarr.get_index('f').values,
						unit=spu.hertz,
						is_sorted=True,
					),
				},
			)

		# T: FluxTime
		if props['monitor_data_type'] == 'FluxTime':
			return ct.InfoFlow(
				dim_names=['t'],
				dim_idx={
					't': ct.ArrayFlow(
						values=xarr.get_index('t').values,
						unit=spu.hertz,
						is_sorted=True,
					),
				},
			)

		# RThetaPhiF: FieldProjectionAngle
		if props['monitor_data_type'] == 'FieldProjectionAngle':
			return ct.InfoFlow(
				dim_names=['r', 'theta', 'phi', 'f'],
				dim_idx={
					'r': ct.ArrayFlow(
						values=xarr.get_index('r').values,
						unit=spu.micrometer,
						is_sorted=True,
					),
				}
				| {
					c: ct.ArrayFlow(
						values=xarr.get_index(c).values, unit=spu.radian, is_sorted=True
					)
					for c in ['r', 'theta', 'phi']
				}
				| {
					'f': ct.ArrayFlow(
						values=xarr.get_index('f').values,
						unit=spu.hertz,
						is_sorted=True,
					),
				},
			)

		# UxUyRF: FieldProjectionKSpace
		if props['monitor_data_type'] == 'FieldProjectionKSpace':
			return ct.InfoFlow(
				dim_names=['ux', 'uy', 'r', 'f'],
				dim_idx={
					c: ct.ArrayFlow(
						values=xarr.get_index(c).values, unit=None, is_sorted=True
					)
					for c in ['ux', 'uy']
				}
				| {
					'r': ct.ArrayFlow(
						values=xarr.get_index('r').values,
						unit=spu.micrometer,
						is_sorted=True,
					),
					'f': ct.ArrayFlow(
						values=xarr.get_index('f').values,
						unit=spu.hertz,
						is_sorted=True,
					),
				},
			)

		# OrderxOrderyF: Diffraction
		if props['monitor_data_type'] == 'Diffraction':
			return ct.InfoFlow(
				dim_names=['orders_x', 'orders_y', 'f'],
				dim_idx={
					f'orders_{c}': ct.ArrayFlow(
						values=xarr.get_index(f'orders_{c}').values,
						unit=None,
						is_sorted=True,
					)
					for c in ['x', 'y']
				}
				| {
					'f': ct.ArrayFlow(
						values=xarr.get_index('f').values,
						unit=spu.hertz,
						is_sorted=True,
					),
				},
			)

		msg = f'Unsupported Monitor Data Type {props["monitor_data_type"]} in "FlowKind.Info" of "{self.bl_label}"'
		raise RuntimeError(msg)


####################
# - Blender Registration
####################
BL_REGISTER = [
	ExtractDataNode,
]
BL_NODES = {ct.NodeType.ExtractData: (ct.NodeCategory.MAXWELLSIM_ANALYSIS)}
