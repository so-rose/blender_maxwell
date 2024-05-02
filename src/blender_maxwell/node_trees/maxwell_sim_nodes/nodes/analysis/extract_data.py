"""Declares `ExtractDataNode`."""

import enum
import typing as typ

import bpy
import jax
import numpy as np
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
		'Monitor Data': {'Monitor Data': sockets.MaxwellMonitorDataSocketDef()},
	}
	output_socket_sets: typ.ClassVar = {
		'Sim Data': {'Monitor Data': sockets.MaxwellMonitorDataSocketDef()},
		'Monitor Data': {'Expr': sockets.ExprSocketDef(active_kind=ct.FlowKind.Array)},
	}

	####################
	# - Properties
	####################
	extract_filter: enum.Enum = bl_cache.BLField(
		prop_ui=True,
		enum_cb=lambda self, _: self.search_extract_filters(),
	)

	####################
	# - Computed: Sim Data
	####################
	@property
	def sim_data(self) -> td.SimulationData | None:
		"""Extracts the simulation data from the input socket.

		Return:
			Either the simulation data, if available, or None.
		"""
		sim_data = self._compute_input(
			'Sim Data', kind=ct.FlowKind.Value, optional=True
		)
		has_sim_data = not ct.FlowSignal.check(sim_data)
		if has_sim_data:
			return sim_data

		return None

	@bl_cache.cached_bl_property()
	def sim_data_monitor_nametype(self) -> dict[str, str] | None:
		"""For simulation data, deduces a map from the monitor name to the monitor "type".

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
		"""Extracts the monitor data from the input socket.

		Return:
			Either the monitor data, if available, or None.
		"""
		monitor_data = self._compute_input(
			'Monitor Data', kind=ct.FlowKind.Value, optional=True
		)
		has_monitor_data = not ct.FlowSignal.check(monitor_data)
		if has_monitor_data:
			return monitor_data

		return None

	@bl_cache.cached_bl_property()
	def monitor_data_type(self) -> str | None:
		r"""For monitor data, deduces the monitor "type".

		- **Field(Time)**: A monitor storing values/pixels/voxels with electromagnetic field values, on the time or frequency domain.
		- **Permittivity**: A monitor storing values/pixels/voxels containing the diagonal of the relative permittivity tensor.
		- **Flux(Time)**: A monitor storing the directional flux on the time or frequency domain.
			For planes, an explicit direction is defined.
			For volumes, the the integral of all outgoing energy is stored.
		- **FieldProjection(...)**: A monitor storing the spherical-coordinate electromagnetic field components of a near-to-far-field projection.
		- **Diffraction**: A monitor storing a near-to-far-field projection by diffraction order.

		Notes:
			Should be invalidated with (before) `self.monitor_data_attrs`.

		Return:
			The "type" of the monitor, if available, else None.
		"""
		if self.monitor_data is not None:
			return self.monitor_data.type.removesuffix('Data')

		return None

	@bl_cache.cached_bl_property()
	def monitor_data_attrs(self) -> list[str] | None:
		r"""For monitor data, deduces the valid data-containing attributes.

		The output depends entirely on the output of `self.monitor_data_type`, since the valid attributes of each monitor type is well-defined without needing to perform dynamic lookups.

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
			Should be reset (via `self.extract_filter`) with (after) `self.sim_data_monitor_nametype`, `self.monitor_data_attrs`, and (implicitly) `self.monitor_type`.

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

		if self.monitor_data_attrs is not None:
			# Field/FieldTime
			if self.monitor_data_type in ['Field', 'FieldTime']:
				return [
					(
						monitor_attr,
						monitor_attr,
						f'ℂ {monitor_attr[1]}-polarization of the {"electric" if monitor_attr[0] == "E" else "magnetic"} field',
						'',
						i,
					)
					for i, monitor_attr in enumerate(self.monitor_data_attrs)
				]

			# Permittivity
			if self.monitor_data_type == 'Permittivity':
				return [
					(monitor_attr, monitor_attr, f'ℂ ε_{monitor_attr}', '', i)
					for i, monitor_attr in enumerate(self.monitor_data_attrs)
				]

			# Flux/FluxTime
			if self.monitor_data_type in ['Flux', 'FluxTime']:
				return [
					(
						monitor_attr,
						monitor_attr,
						'Power flux integral through the plane / out of the volume',
						'',
						i,
					)
					for i, monitor_attr in enumerate(self.monitor_data_attrs)
				]

			# FieldProjection(Angle/Cartesian/KSpace)/Diffraction
			if self.monitor_data_type in [
				'FieldProjectionAngle',
				'FieldProjectionCartesian',
				'FieldProjectionKSpace',
				'Diffraction',
			]:
				return [
					(
						monitor_attr,
						monitor_attr,
						f'ℂ {monitor_attr[1]}-component of the spherical {"electric" if monitor_attr[0] == "E" else "magnetic"} field',
						'',
						i,
					)
					for i, monitor_attr in enumerate(self.monitor_data_attrs)
				]

		return []

	####################
	# - UI
	####################
	def draw_label(self) -> None:
		"""Show the extracted data (if any) in the node's header label.

		Notes:
			Called by Blender to determine the text to place in the node's header.
		"""
		has_sim_data = self.sim_data_monitor_nametype is not None
		has_monitor_data = self.monitor_data_attrs is not None

		if has_sim_data or has_monitor_data:
			return f'Extract: {self.extract_filter}'

		return self.bl_label

	def draw_props(self, _: bpy.types.Context, col: bpy.types.UILayout) -> None:
		"""Draw node properties in the node.

		Parameters:
			col: UI target for drawing.
		"""
		col.prop(self, self.blfields['extract_filter'], text='')

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
		self.monitor_data_attrs = bl_cache.Signal.InvalidateCache
		self.extract_filter = bl_cache.Signal.ResetEnumItems

	####################
	# - Output (Value): Sim Data -> Monitor Data
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
		"""Compute `Monitor Data` by querying the attribute of `Sim Data` referenced by the property `self.extract_filter`.

		Returns:
			Monitor data, if available, else `ct.FlowSignal.FlowPending`.
		"""
		extract_filter = props['extract_filter']
		sim_data = input_sockets['Sim Data']
		has_sim_data = not ct.FlowSignal.check(sim_data)

		if has_sim_data and extract_filter is not None:
			return sim_data.monitor_data[extract_filter]

		return ct.FlowSignal.FlowPending

	####################
	# - Output (Array): Monitor Data -> Expr
	####################
	@events.computes_output_socket(
		# Trigger
		'Expr',
		kind=ct.FlowKind.Array,
		# Loaded
		props={'extract_filter'},
		input_sockets={'Monitor Data'},
		input_socket_kinds={'Monitor Data': ct.FlowKind.Value},
	)
	def compute_expr(
		self, props: dict, input_sockets: dict
	) -> jax.Array | ct.FlowSignal:
		"""Compute `Expr:Array` by querying an array-like attribute of `Monitor Data`, then constructing an `ct.ArrayFlow` around it.

		Uses the internal `xarray` data returned by Tidy3D.
		By using `np.array` on the `.data` attribute of the `xarray`, instead of the usual JAX array constructor, we should save a (possibly very big) copy.

		Returns:
			The data array, if available, else `ct.FlowSignal.FlowPending`.
		"""
		extract_filter = props['extract_filter']
		monitor_data = input_sockets['Monitor Data']
		has_monitor_data = not ct.FlowSignal.check(monitor_data)

		if has_monitor_data and extract_filter is not None:
			xarray_data = getattr(monitor_data, extract_filter)
			return ct.ArrayFlow(values=np.array(xarray_data.data), unit=None)

		return ct.FlowSignal.FlowPending

	@events.computes_output_socket(
		# Trigger
		'Expr',
		kind=ct.FlowKind.LazyValueFunc,
		# Loaded
		output_sockets={'Expr'},
		output_socket_kinds={'Expr': ct.FlowKind.Array},
	)
	def compute_extracted_data_lazy(
		self, output_sockets: dict
	) -> ct.LazyValueFuncFlow | None:
		"""Declare `Expr:LazyValueFunc` by creating a simple function that directly wraps `Expr:Array`.

		Returns:
			The composable function array, if available, else `ct.FlowSignal.FlowPending`.
		"""
		output_expr = output_sockets['Expr']
		has_output_expr = not ct.FlowSignal.check(output_expr)

		if has_output_expr:
			return ct.LazyValueFuncFlow(
				func=lambda: output_expr.values, supports_jax=True
			)

		return ct.FlowSignal.FlowPending

	####################
	# - Auxiliary (Params): Monitor Data -> Expr
	####################
	@events.computes_output_socket(
		'Expr',
		kind=ct.FlowKind.Params,
	)
	def compute_data_params(self) -> ct.ParamsFlow:
		"""Declare an empty `Data:Params`, to indicate the start of a function-composition pipeline.

		Returns:
			A completely empty `ParamsFlow`, ready to be composed.
		"""
		return ct.ParamsFlow()

	####################
	# - Auxiliary (Info): Monitor Data -> Expr
	####################
	@events.computes_output_socket(
		# Trigger
		'Expr',
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
		monitor_data = input_sockets['Monitor Data']
		monitor_data_type = props['monitor_data_type']
		extract_filter = props['extract_filter']

		has_monitor_data = not ct.FlowSignal.check(monitor_data)

		# Retrieve XArray
		if has_monitor_data and extract_filter is not None:
			xarr = getattr(monitor_data, extract_filter)
		else:
			return ct.FlowSignal.FlowPending

		# Compute InfoFlow from XArray
		## XYZF: Field / Permittivity / FieldProjectionCartesian
		if monitor_data_type in {
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
				output_name=extract_filter,
				output_shape=None,
				output_mathtype=spux.MathType.Complex,
				output_unit=(
					spu.volt / spu.micrometer if monitor_data_type == 'Field' else None
				),
			)

		## XYZT: FieldTime
		if monitor_data_type == 'FieldTime':
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
				output_name=extract_filter,
				output_shape=None,
				output_mathtype=spux.MathType.Complex,
				output_unit=(
					spu.volt / spu.micrometer if monitor_data_type == 'Field' else None
				),
			)

		## F: Flux
		if monitor_data_type == 'Flux':
			return ct.InfoFlow(
				dim_names=['f'],
				dim_idx={
					'f': ct.ArrayFlow(
						values=xarr.get_index('f').values,
						unit=spu.hertz,
						is_sorted=True,
					),
				},
				output_name=extract_filter,
				output_shape=None,
				output_mathtype=spux.MathType.Real,
				output_unit=spu.watt,
			)

		## T: FluxTime
		if monitor_data_type == 'FluxTime':
			return ct.InfoFlow(
				dim_names=['t'],
				dim_idx={
					't': ct.ArrayFlow(
						values=xarr.get_index('t').values,
						unit=spu.hertz,
						is_sorted=True,
					),
				},
				output_name=extract_filter,
				output_shape=None,
				output_mathtype=spux.MathType.Real,
				output_unit=spu.watt,
			)

		## RThetaPhiF: FieldProjectionAngle
		if monitor_data_type == 'FieldProjectionAngle':
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
				output_name=extract_filter,
				output_shape=None,
				output_mathtype=spux.MathType.Real,
				output_unit=(
					spu.volt / spu.micrometer
					if extract_filter.startswith('E')
					else spu.ampere / spu.micrometer
				),
			)

		## UxUyRF: FieldProjectionKSpace
		if monitor_data_type == 'FieldProjectionKSpace':
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
				output_name=extract_filter,
				output_shape=None,
				output_mathtype=spux.MathType.Real,
				output_unit=(
					spu.volt / spu.micrometer
					if extract_filter.startswith('E')
					else spu.ampere / spu.micrometer
				),
			)

		## OrderxOrderyF: Diffraction
		if monitor_data_type == 'Diffraction':
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
				output_name=extract_filter,
				output_shape=None,
				output_mathtype=spux.MathType.Real,
				output_unit=(
					spu.volt / spu.micrometer
					if extract_filter.startswith('E')
					else spu.ampere / spu.micrometer
				),
			)

		msg = f'Unsupported Monitor Data Type {monitor_data_type} in "FlowKind.Info" of "{self.bl_label}"'
		raise RuntimeError(msg)


####################
# - Blender Registration
####################
BL_REGISTER = [
	ExtractDataNode,
]
BL_NODES = {ct.NodeType.ExtractData: (ct.NodeCategory.MAXWELLSIM_ANALYSIS)}
