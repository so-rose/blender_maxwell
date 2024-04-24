import enum
import typing as typ

import bpy
import jax
import jax.numpy as jnp
import sympy.physics.units as spu
import tidy3d as td

from blender_maxwell.utils import bl_cache, logger
from blender_maxwell.utils import extra_sympy_units as spux

from ... import contracts as ct
from ... import sockets
from .. import base, events

log = logger.get(__name__)

TDMonitorData: typ.TypeAlias = td.components.data.monitor_data.MonitorData


class ExtractDataNode(base.MaxwellSimNode):
	"""Extract data from sockets for further analysis.

	# Socket Sets
	## Sim Data
	Extracts monitors from a `MaxwelFDTDSimDataSocket`.

	## Monitor Data
	Extracts array attributes from a `MaxwelFDTDSimDataSocket`.

	Attributes:
		extract_filter: Identifier for data to extract from the input.

	"""

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
	extract_filter: enum.Enum = bl_cache.BLField(
		None,
		prop_ui=True,
		enum_cb=lambda self, _: self.search_extract_filters(),
	)

	####################
	# - Computed: Sim Data
	####################
	@property
	def sim_data(self) -> td.SimulationData | None:
		"""Computes the (cached) simulation data from the input socket.

		Return:
			Either the simulation data, if available, or None.
		"""
		sim_data = self._compute_input(
			'Sim Data', kind=ct.FlowKind.Value, optional=True
		)
		if not ct.FlowSignal.check(sim_data):
			return sim_data

		return None

	@bl_cache.cached_bl_property()
	def sim_data_monitor_nametype(self) -> dict[str, str] | None:
		"""For simulation data, computes and and caches a map from name to "type".

		Return:
			The name to type of monitors in the simulation data.
		"""
		if self.sim_data is not None:
			return {
				monitor_name: monitor_data.type
				for monitor_name, monitor_data in self.sim_data.monitor_data.items()
			}

		return None

	####################
	# - Computed Properties: Monitor Data
	####################
	@property
	def monitor_data(self) -> TDMonitorData | None:
		"""Computes the (cached) monitor data from the input socket.

		Return:
			Either the monitor data, if available, or None.
		"""
		monitor_data = self._compute_input(
			'Monitor Data', kind=ct.FlowKind.Value, optional=True
		)
		if not ct.FlowSignal.check(monitor_data):
			return monitor_data

		return None

	@bl_cache.cached_bl_property()
	def monitor_data_type(self) -> str | None:
		"""For monitor data, computes and caches the monitor "type".

		Notes:
			Should be invalidated with (before) `self.monitor_data_components`.

		Return:
			The "type" of the monitor, if available, else None.
		"""
		if self.monitor_data is not None:
			return self.monitor_data.type.removesuffix('Data')

		return None

	@bl_cache.cached_bl_property()
	def monitor_data_components(self) -> list[str] | None:
		r"""For monitor data, computes and caches the component sof the monitor.

		The output depends entirely on the output of `self.monitor_data`.

		- **Field(Time)**: Whichever `[E|H][x|y|z]` are not `None` on the monitor.
		- **Permittivity**: Specifically `['xx', 'yy', 'zz']`.
		- **Flux(Time)**: Only `['flux']`.
		- **FieldProjection(...)**: All of $r$, $\theta$, $\phi$ for both `E` and `H`.
		- **Diffraction**: Same as `FieldProjection`.

		Notes:
			Should be invalidated after with `self.monitor_data_type`.

		Return:
			The "type" of the monitor, if available, else None.
		"""
		if self.monitor_data is not None:
			# Field/FieldTime
			if self.monitor_data_type in ['Field', 'FieldTime']:
				return [
					field_component
					for field_component in ['Ex', 'Ey', 'Ez', 'Hx', 'Hy', 'Hz']
					if hasattr(self.monitor_data, field_component)
				]

			# Permittivity
			if self.monitor_data_type == 'Permittivity':
				return ['xx', 'yy', 'zz']

			# Flux/FluxTime
			if self.monitor_data_type in ['Flux', 'FluxTime']:
				return ['flux']

			# FieldProjection(Angle/Cartesian/KSpace)/Diffraction
			if self.monitor_data_type in [
				'FieldProjectionAngle',
				'FieldProjectionCartesian',
				'FieldProjectionKSpace',
				'Diffraction',
			]:
				return [
					'Er',
					'Etheta',
					'Ephi',
					'Hr',
					'Htheta',
					'Hphi',
				]

		return None

	####################
	# - Extraction Filter Search
	####################
	def search_extract_filters(self) -> list[ct.BLEnumElement]:
		"""Compute valid values for `self.extract_filter`, for a dynamic `EnumProperty`.

		Notes:
			Should be reset (via `self.extract_filter`) with (after) `self.sim_data_monitor_nametype`, `self.monitor_data_components`, and (implicitly) `self.monitor_type`.

			See `bl_cache.BLField` for more on dynamic `EnumProperty`.

		Returns:
			Valid `self.extract_filter` in a format compatible with dynamic `EnumProperty`.
		"""
		if self.sim_data_monitor_nametype is not None:
			return [
				(monitor_name, monitor_name, monitor_type.removesuffix('Data'), '', i)
				for i, (monitor_name, monitor_type) in enumerate(
					self.sim_data_monitor_nametype.items()
				)
			]

		if self.monitor_data_components is not None:
			return [
				(
					component_name,
					component_name,
					f'ℂ {component_name[1]}-polarization of the {"electric" if component_name[0] == "E" else "magnetic"} field',
					'',
					i,
				)
				for i, component_name in enumerate(self.monitor_data_components)
			]

		return []

	####################
	# - UI
	####################
	def draw_props(self, _: bpy.types.Context, col: bpy.types.UILayout) -> None:
		"""Draw node properties in the node.

		Parameters:
			col: UI target for drawing.
		"""
		col.prop(self, self.blfields['extract_filter'], text='')

	def draw_info(self, _: bpy.types.Context, col: bpy.types.UILayout) -> None:
		"""Draw dynamic information in the node, for user consideration.

		Parameters:
			col: UI target for drawing.
		"""
		has_sim_data = self.sim_data_monitor_nametype is not None
		has_monitor_data = self.monitor_data_components is not None

		if has_sim_data or has_monitor_data:
			# Header
			row = col.row()
			row.alignment = 'CENTER'
			if has_sim_data:
				row.label(text=f'{len(self.sim_data_monitor_nametype)} Monitors')
			elif has_monitor_data:
				row.label(text=f'{self.monitor_data_type} Monitor Data')

			# Monitor Data Contents
			## TODO: More compact double-split
			## TODO: Output shape data.
			## TODO: Local ENUM_MANY tabs for visible column selection?
			row = col.row()
			box = row.box()
			grid = box.grid_flow(row_major=True, columns=2, even_columns=True)
			for monitor_name, monitor_type in self.sim_data_monitor_nametype.items():
				grid.label(text=monitor_name)
				grid.label(text=monitor_type)

	####################
	# - Events
	####################
	@events.on_value_changed(
		# Trigger
		socket_name={'Sim Data', 'Monitor Data'},
		prop_name='active_socket_set',
		run_on_init=True,
	)
	def on_input_sockets_changed(self) -> None:
		"""Invalidate the cached properties for sim data / monitor data, and reset the extraction filter."""
		self.sim_data_monitor_nametype = bl_cache.Signal.InvalidateCache
		self.monitor_data_type = bl_cache.Signal.InvalidateCache
		self.monitor_data_components = bl_cache.Signal.InvalidateCache
		self.extract_filter = bl_cache.Signal.ResetEnumItems

	####################
	# - Output: Sim Data -> Monitor Data
	####################
	@events.computes_output_socket(
		# Trigger
		'Monitor Data',
		kind=ct.FlowKind.Value,
		# Loaded
		props={'extract_filter'},
		input_sockets={'Sim Data'},
	)
	def compute_monitor_data(
		self, props: dict, input_sockets: dict
	) -> TDMonitorData | ct.FlowSignal:
		"""Compute `Monitor Data` by querying an attribute of `Sim Data`.

		Notes:
			The attribute to query is read directly from `self.extract_filter`.
			This is also the mechanism that protects from trying to reference an invalid attribute.

		Returns:
			Monitor data, if available, else `ct.FlowSignal.FlowPending`.
		"""
		sim_data = input_sockets['Sim Data']
		has_sim_data = not ct.FlowSignal.check(sim_data)

		if has_sim_data and props['extract_filter'] != 'NONE':
			return input_sockets['Sim Data'].monitor_data[props['extract_filter']]

		# Propagate NoFlow
		if ct.FlowSignal.check_single(sim_data, ct.FlowSignal.NoFlow):
			return ct.FlowSignal.NoFlow

		return ct.FlowSignal.FlowPending

	####################
	# - Output: Monitor Data -> Data
	####################
	@events.computes_output_socket(
		# Trigger
		'Data',
		kind=ct.FlowKind.Array,
		# Loaded
		props={'extract_filter'},
		input_sockets={'Monitor Data'},
		input_socket_kinds={'Monitor Data': ct.FlowKind.Value},
	)
	def compute_data(
		self, props: dict, input_sockets: dict
	) -> jax.Array | ct.FlowSignal:
		"""Compute `Data:Array` by querying an array-like attribute of `Monitor Data`, then constructing an `ct.ArrayFlow`.

		Uses the internal `xarray` data returned by Tidy3D.

		Notes:
			The attribute to query is read directly from `self.extract_filter`.
			This is also the mechanism that protects from trying to reference an invalid attribute.

			Used as the first part of the `LazyFuncValue` chain used for further array manipulations with Math nodes.

		Returns:
			The data array, if available, else `ct.FlowSignal.FlowPending`.
		"""
		has_monitor_data = not ct.FlowSignal.check(input_sockets['Monitor Data'])

		if has_monitor_data and props['extract_filter'] != 'NONE':
			xarray_data = getattr(
				input_sockets['Monitor Data'], props['extract_filter']
			)
			return ct.ArrayFlow(values=jnp.array(xarray_data.data), unit=None)
			## TODO: Try np.array instead, as it removes a copy, while still (I believe) being JIT-compatible.

		return ct.FlowSignal.FlowPending

	@events.computes_output_socket(
		# Trigger
		'Data',
		kind=ct.FlowKind.LazyValueFunc,
		# Loaded
		output_sockets={'Data'},
		output_socket_kinds={'Data': ct.FlowKind.Array},
	)
	def compute_extracted_data_lazy(
		self, output_sockets: dict
	) -> ct.LazyValueFuncFlow | None:
		"""Declare `Data:LazyValueFunc` by creating a simple function that directly wraps `Data:Array`.

		Returns:
			The composable function array, if available, else `ct.FlowSignal.FlowPending`.
		"""
		has_output_data = not ct.FlowSignal.check(output_sockets['Data'])

		if has_output_data:
			return ct.LazyValueFuncFlow(
				func=lambda: output_sockets['Data'].values, supports_jax=True
			)

		return ct.FlowSignal.FlowPending

	####################
	# - Auxiliary: Monitor Data -> Data
	####################
	@events.computes_output_socket(
		'Data',
		kind=ct.FlowKind.Params,
	)
	def compute_data_params(self) -> ct.ParamsFlow:
		return ct.ParamsFlow()

	@events.computes_output_socket(
		# Trigger
		'Data',
		kind=ct.FlowKind.Info,
		# Loaded
		props={'monitor_data_type', 'extract_filter'},
		input_sockets={'Monitor Data'},
		input_socket_kinds={'Monitor Data': ct.FlowKind.Value},
		input_sockets_optional={'Monitor Data': True},
	)
	def compute_extracted_data_info(
		self, props: dict, input_sockets: dict
	) -> ct.InfoFlow:
		"""Declare `Data:Info` by manually selecting appropriate axes, units, etc. for each monitor type.

		Returns:
			Information describing the `Data:LazyValueFunc`, if available, else `ct.FlowSignal.FlowPending`.
		"""
		has_monitor_data = not ct.FlowSignal.check(input_sockets['Monitor Data'])

		# Retrieve XArray
		if has_monitor_data and props['extract_filter'] != 'NONE':
			xarr = getattr(input_sockets['Monitor Data'], props['extract_filter'])
		else:
			return ct.FlowSignal.FlowPending

		info_output_names = {
			'output_names': [props['extract_filter']],
		}

		# Compute InfoFlow from XArray
		## XYZF: Field / Permittivity / FieldProjectionCartesian
		if props['monitor_data_type'] in {
			'Field',
			'Permittivity',
			#'FieldProjectionCartesian',
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
				**info_output_names,
				output_mathtypes={props['extract_filter']: spux.MathType.Complex},
				output_units={
					props['extract_filter']: spu.volt / spu.micrometer
					if props['monitor_data_type'] == 'Field'
					else None
				},
			)

		## XYZT: FieldTime
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
				**info_output_names,
				output_mathtypes={props['extract_filter']: spux.MathType.Complex},
				output_units={
					props['extract_filter']: (
						spu.volt / spu.micrometer
						if props['extract_filter'].startswith('E')
						else spu.ampere / spu.micrometer
					)
					if props['monitor_data_type'] == 'Field'
					else None
				},
			)

		## F: Flux
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
				**info_output_names,
				output_mathtypes={props['extract_filter']: spux.MathType.Real},
				output_units={props['extract_filter']: spu.watt},
			)

		## T: FluxTime
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
				**info_output_names,
				output_mathtypes={props['extract_filter']: spux.MathType.Real},
				output_units={props['extract_filter']: spu.watt},
			)

		## RThetaPhiF: FieldProjectionAngle
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
				**info_output_names,
				output_mathtypes={props['extract_filter']: spux.MathType.Real},
				output_units={
					props['extract_filter']: (
						spu.volt / spu.micrometer
						if props['extract_filter'].startswith('E')
						else spu.ampere / spu.micrometer
					)
				},
			)

		## UxUyRF: FieldProjectionKSpace
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
				**info_output_names,
				output_mathtypes={props['extract_filter']: spux.MathType.Real},
				output_units={
					props['extract_filter']: (
						spu.volt / spu.micrometer
						if props['extract_filter'].startswith('E')
						else spu.ampere / spu.micrometer
					)
				},
			)

		## OrderxOrderyF: Diffraction
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
				**info_output_names,
				output_mathtypes={props['extract_filter']: spux.MathType.Real},
				output_units={
					props['extract_filter']: (
						spu.volt / spu.micrometer
						if props['extract_filter'].startswith('E')
						else spu.ampere / spu.micrometer
					)
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
