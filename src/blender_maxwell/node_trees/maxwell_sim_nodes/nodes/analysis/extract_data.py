# blender_maxwell
# Copyright (C) 2024 blender_maxwell Project Contributors
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""Implements `ExtractDataNode`."""

import enum
import functools
import typing as typ

import bpy
import jax.numpy as jnp
import sympy.physics.units as spu
import tidy3d as td
import xarray

from blender_maxwell.utils import bl_cache, logger, sim_symbols

from ... import contracts as ct
from ... import sockets
from .. import base, events

log = logger.get(__name__)

TDMonitorData: typ.TypeAlias = td.components.data.monitor_data.MonitorData

RealizedSymsVals: typ.TypeAlias = tuple[sim_symbols.SimSymbol, ...], tuple[typ.Any, ...]
SimDataArray: typ.TypeAlias = dict[RealizedSymsVals, td.SimulationData]
SimDataArrayInfo: typ.TypeAlias = dict[RealizedSymsVals, typ.Any]

FK = ct.FlowKind
FS = ct.FlowSignal


####################
# - Monitor Labelling
####################
def valid_monitor_attrs(
	example_sim_data: td.SimulationData, monitor_name: str
) -> tuple[str, ...]:
	"""Retrieve the valid attributes of `sim_data.monitor_data' from a valid `sim_data` of type `td.SimulationData`.

	Parameters:
		monitor_type: The name of the monitor type, with the 'Data' prefix removed.
	"""
	monitor_data = example_sim_data.monitor_data[monitor_name]
	monitor_type = monitor_data.type.removesuffix('Data')

	match monitor_type:
		case 'Field' | 'FieldTime' | 'Mode':
			## TODO: flux, poynting, intensity
			return tuple(
				[
					field_component
					for field_component in ['Ex', 'Ey', 'Ez', 'Hx', 'Hy', 'Hz']
					if getattr(monitor_data, field_component, None) is not None
				]
			)

		case 'Permittivity':
			return ('eps_xx', 'eps_yy', 'eps_zz')

		case 'Flux' | 'FluxTime':
			return ('flux',)

		case (
			'FieldProjectionAngle'
			| 'FieldProjectionCartesian'
			| 'FieldProjectionKSpace'
			| 'Diffraction'
		):
			return (
				'Er',
				'Etheta',
				'Ephi',
				'Hr',
				'Htheta',
				'Hphi',
			)

	raise TypeError


####################
# - Extract InfoFlow
####################
MONITOR_SYMBOLS: dict[str, sim_symbols.SimSymbol] = {
	# Field Label
	'EH*': sim_symbols.sim_axis_idx(None),
	# Cartesian
	'x': sim_symbols.space_x(spu.micrometer),
	'y': sim_symbols.space_y(spu.micrometer),
	'z': sim_symbols.space_z(spu.micrometer),
	# Spherical
	'r': sim_symbols.ang_r(spu.micrometer),
	'theta': sim_symbols.ang_theta(spu.radian),
	'phi': sim_symbols.ang_phi(spu.radian),
	# Freq|Time
	'f': sim_symbols.freq(spu.hertz),
	't': sim_symbols.t(spu.second),
	# Power Flux
	'flux': sim_symbols.flux(spu.watt),
	# Wavevector
	'ux': sim_symbols.dir_x(spu.watt),
	'uy': sim_symbols.dir_y(spu.watt),
	# Diffraction Orders
	'orders_x': sim_symbols.diff_order_x(None),
	'orders_y': sim_symbols.diff_order_y(None),
	# Cartesian Fields
	'field': sim_symbols.field_e(spu.volt / spu.micrometer),  ## TODO: H???
	'field_e': sim_symbols.field_e(spu.volt / spu.micrometer),
	'field_h': sim_symbols.field_h(spu.ampere / spu.micrometer),
}


def _mk_idx_array(xarr: xarray.DataArray, axis: str) -> ct.RangeFlow | ct.ArrayFlow:
	return ct.RangeFlow.try_from_array(
		ct.ArrayFlow(
			jax_bytes=xarr.get_index(axis).values,
			unit=MONITOR_SYMBOLS[axis].unit,
			is_sorted=True,
		)
	)


def output_symbol_by_type(monitor_type: str) -> sim_symbols.SimSymbol:
	match monitor_type:
		case 'Field' | 'FieldProjectionCartesian' | 'Permittivity' | 'Mode':
			return MONITOR_SYMBOLS['field_e']

		case 'FieldTime':
			return MONITOR_SYMBOLS['field']

		case 'Flux':
			return MONITOR_SYMBOLS['flux']

		case 'FluxTime':
			return MONITOR_SYMBOLS['flux']

		case 'FieldProjectionAngle':
			return MONITOR_SYMBOLS['field']

		case 'FieldProjectionKSpace':
			return MONITOR_SYMBOLS['field']

		case 'Diffraction':
			return MONITOR_SYMBOLS['field']

	return None


def _extract_info(
	example_xarr: xarray.DataArray,
	monitor_type: str,
	monitor_attrs: tuple[str, ...],
	batch_dims: dict[sim_symbols.SimSymbol, ct.RangeFlow | ct.ArrayFlow],
) -> ct.InfoFlow | None:
	log.debug([monitor_type, monitor_attrs, batch_dims])
	mk_idx_array = functools.partial(_mk_idx_array, example_xarr)
	match monitor_type:
		case 'Field' | 'FieldProjectionCartesian' | 'Permittivity' | 'Mode':
			return ct.InfoFlow(
				dims=batch_dims
				| {
					MONITOR_SYMBOLS['EH*']: monitor_attrs,
					MONITOR_SYMBOLS['x']: mk_idx_array('x'),
					MONITOR_SYMBOLS['y']: mk_idx_array('y'),
					MONITOR_SYMBOLS['z']: mk_idx_array('z'),
					MONITOR_SYMBOLS['f']: mk_idx_array('f'),
				},
				output=MONITOR_SYMBOLS['field_e'],
			)

		case 'FieldTime':
			return ct.InfoFlow(
				dims=batch_dims
				| {
					MONITOR_SYMBOLS['EH*']: monitor_attrs,
					MONITOR_SYMBOLS['x']: mk_idx_array('x'),
					MONITOR_SYMBOLS['y']: mk_idx_array('y'),
					MONITOR_SYMBOLS['z']: mk_idx_array('z'),
					MONITOR_SYMBOLS['t']: mk_idx_array('t'),
				},
				output=MONITOR_SYMBOLS['field'],
			)

		case 'Flux':
			return ct.InfoFlow(
				dims=batch_dims
				| {
					MONITOR_SYMBOLS['f']: mk_idx_array('f'),
				},
				output=MONITOR_SYMBOLS['flux'],
			)

		case 'FluxTime':
			return ct.InfoFlow(
				dims=batch_dims
				| {
					MONITOR_SYMBOLS['t']: mk_idx_array('t'),
				},
				output=MONITOR_SYMBOLS['flux'],
			)

		case 'FieldProjectionAngle':
			return ct.InfoFlow(
				dims=batch_dims
				| {
					MONITOR_SYMBOLS['EH*']: monitor_attrs,
					MONITOR_SYMBOLS['r']: mk_idx_array('r'),
					MONITOR_SYMBOLS['theta']: mk_idx_array('theta'),
					MONITOR_SYMBOLS['phi']: mk_idx_array('phi'),
					MONITOR_SYMBOLS['f']: mk_idx_array('f'),
				},
				output=MONITOR_SYMBOLS['field'],
			)

		case 'FieldProjectionKSpace':
			return ct.InfoFlow(
				dims=batch_dims
				| {
					MONITOR_SYMBOLS['EH*']: monitor_attrs,
					MONITOR_SYMBOLS['ux']: mk_idx_array('ux'),
					MONITOR_SYMBOLS['uy']: mk_idx_array('uy'),
					MONITOR_SYMBOLS['r']: mk_idx_array('r'),
					MONITOR_SYMBOLS['f']: mk_idx_array('f'),
				},
				output=MONITOR_SYMBOLS['field'],
			)

		case 'Diffraction':
			return ct.InfoFlow(
				dims=batch_dims
				| {
					MONITOR_SYMBOLS['EH*']: monitor_attrs,
					MONITOR_SYMBOLS['orders_x']: mk_idx_array('orders_x'),
					MONITOR_SYMBOLS['orders_y']: mk_idx_array('orders_y'),
					MONITOR_SYMBOLS['f']: mk_idx_array('f'),
				},
				output=MONITOR_SYMBOLS['field'],
			)

	raise TypeError


def extract_monitor_xarrs(
	monitor_datas: dict[RealizedSymsVals, typ.Any], monitor_attrs: tuple[str, ...]
) -> dict[RealizedSymsVals, ct.InfoFlow]:
	return {
		syms_vals: {
			monitor_attr: getattr(monitor_data, monitor_attr, None)
			for monitor_attr in monitor_attrs
		}
		for syms_vals, monitor_data in monitor_datas.items()
	}


def extract_info(
	monitor_datas: dict[RealizedSymsVals, typ.Any], monitor_attrs: tuple[str, ...]
) -> dict[RealizedSymsVals, ct.InfoFlow]:
	"""Extract an InfoFlow describing monitor data from a batch of simulations."""
	# Extract Dimension from Batched Values
	## -> Comb the data to expose each symbol's realized values as an array.
	## -> Each symbol: array can then become a dimension.
	## -> These are the "batch dimensions", which allows indexing across sims.
	## -> The retained sim symbol provides semantic index coordinates.
	example_syms_vals = next(iter(monitor_datas.keys()))
	syms = example_syms_vals[0] if example_syms_vals != () else ()
	vals_per_sym_pos = (
		[vals for _, vals in monitor_datas] if example_syms_vals != () else []
	)

	_batch_dims = dict(
		zip(
			syms,
			zip(*vals_per_sym_pos, strict=True),
			strict=True,
		)
	)
	batch_dims = {
		sym: ct.RangeFlow.try_from_array(
			ct.ArrayFlow(
				jax_bytes=vals,
				unit=sym.unit,
				is_sorted=True,
			)
		)
		for sym, vals in _batch_dims.items()
	}

	# Extract Example Monitor Data | XArray
	## -> We presume all monitor attributes have the exact same dims + output.
	## -> Because of this, we only need one "example" xarray.
	## -> This xarray will be used to extract dimensional coordinates...
	## -> ...Presuming that these coords will generalize.
	example_monitor_data = next(iter(monitor_datas.values()))
	monitor_datas_xarrs = extract_monitor_xarrs(monitor_datas, monitor_attrs)

	# Extract XArray for Each Monitor Attribute
	example_monitor_data_xarrs = next(iter(monitor_datas_xarrs.values()))
	example_xarr = next(iter(example_monitor_data_xarrs.values()))

	# Extract InfoFlow of First
	## -> All of the InfoFlows should be identical...
	## -> ...Apart from the batched dimensions.
	return _extract_info(
		example_xarr,
		example_monitor_data.type.removesuffix('Data'),
		monitor_attrs,
		batch_dims,
	)


####################
# - Node
####################
class ExtractDataNode(base.MaxwellSimNode):
	"""Extract data from sockets for further analysis.

	Socket Sets:
		Sim Data: Extract monitor data from simulation data by-name.
		Monitor Data: Extract `Expr`s from monitor data by-component.

	Attributes:
		monitor_attr: Identifier for data to extract from the input.
	"""

	node_type = ct.NodeType.ExtractData
	bl_label = 'Extract'

	input_socket_sets: typ.ClassVar = {
		'Single': {
			'Sim Data': sockets.MaxwellFDTDSimDataSocketDef(),
		},
		'Batch': {
			'Sim Datas': sockets.MaxwellFDTDSimDataSocketDef(active_kind=FK.Array),
		},
	}
	# output_sockets: typ.ClassVar = {
	# 'Expr': sockets.ExprSocketDef(active_kind=FK.Func),
	# }
	output_socket_sets: typ.ClassVar = {
		'Single': {
			'Expr': sockets.ExprSocketDef(active_kind=FK.Func),
			'Log': sockets.StringSocketDef(),
		},
		'Batch': {
			'Expr': sockets.ExprSocketDef(active_kind=FK.Func),
			'Logs': sockets.StringSocketDef(active_kind=FK.Array),
		},
	}

	####################
	# - Properties: Sim Datas
	####################
	@events.on_value_changed(
		socket_name={'Sim Data': FK.Value, 'Sim Datas': FK.Array},
	)
	def on_sim_datas_changed(self) -> None:  # noqa: D102
		self.sim_datas = bl_cache.Signal.InvalidateCache

	@bl_cache.cached_bl_property(depends_on={'active_socket_set'})
	def sim_datas(self) -> list[td.SimulationData] | None:
		"""Extracts the simulation data from the input socket.

		Return:
			Either the simulation data, if available, or None.
		"""
		## TODO: Check that syms are identical for all (aka. that we have a batch)
		if self.active_socket_set == 'Single':
			sim_data = self._compute_input('Sim Data', kind=FK.Value)
			has_sim_data = not FS.check(sim_data)
			if has_sim_data:
				# Embedded Symbolic Realizations
				## -> ['realizations'] contains a 2-tuple
				## -> First should be the dict-dump of a SimSymbol.
				## -> Second should be either a value, or a list of values.
				if 'realizations' in sim_data.attrs:
					raw_realizations = sim_data.attrs['realizations']
					syms_vals = {
						sim_symbols.SimSymbol(**raw_sym): raw_val
						if not isinstance(raw_val, tuple | list)
						else jnp.array(raw_val)
						for raw_sym, raw_val in raw_realizations
					}
					return {syms_vals: sim_data}

				# No Embedded Realizations
				return {(): sim_data}

		if self.active_socket_set == 'Batch':
			_sim_datas = self._compute_input('Sim Datas', kind=FK.Value)
			has_sim_datas = not FS.check(_sim_datas)
			if has_sim_datas:
				sim_datas = {}
				for sim_data in sim_datas:
					# Embedded Symbolic Realizations
					## -> ['realizations'] contains a 2-tuple
					## -> First should be the dict-dump of a SimSymbol.
					## -> Second should be either a value, or a list of values.
					if 'realizations' in sim_data.attrs:
						raw_realizations = sim_data.attrs['realizations']
						syms = {
							sim_symbols.SimSymbol(**raw_sym): raw_val
							if not isinstance(raw_val, tuple | list)
							else jnp.array(raw_val)
							for raw_sym, raw_val in raw_realizations
						}
						sim_datas |= {syms_vals: sim_data}

					# No Embedded Realizations
					sim_datas |= {(): sim_data}

		return None

	@bl_cache.cached_bl_property(depends_on={'sim_datas'})
	def example_sim_data(self) -> list[td.SimulationData] | None:
		"""Extracts a single, example simulation data from the input socket.

		All simulation datas share certain properties, ex. names and types of monitors.
		Therefore, we may often only need an example simulation data object.

		Return:
			Either the simulation data, if available, or None.
		"""
		if self.sim_datas:
			return next(iter(self.sim_datas.values()))
		return None

	####################
	# - Properties: Monitor Name
	####################
	@bl_cache.cached_bl_property(depends_on={'example_sim_data'})
	def monitor_types(self) -> dict[str, str] | None:
		"""Dictionary from monitor names on `self.sim_datas` to their associated type name (with suffix 'Data' removed).

		Return:
			The name to type of monitors in the simulation data.
		"""
		if self.example_sim_data is not None:
			return {
				monitor_name: monitor_data.type.removesuffix('Data')
				for monitor_name, monitor_data in self.example_sim_data.monitor_data.items()
			}

		return None

	monitor_name: enum.StrEnum = bl_cache.BLField(
		enum_cb=lambda self, _: self.search_monitor_names(),
		cb_depends_on={'monitor_types'},
	)

	def search_monitor_names(self) -> list[ct.BLEnumElement]:
		"""Compute valid values for `self.monitor_attr`, for a dynamic `EnumProperty`.

		Notes:
			Should be reset (via `self.monitor_attr`) with (after) `self.sim_data_monitor_nametypes`, `self.monitor_data_attrs`, and (implicitly) `self.monitor_type`.

			See `bl_cache.BLField` for more on dynamic `EnumProperty`.

		Returns:
			Valid `self.monitor_attr` in a format compatible with dynamic `EnumProperty`.
		"""
		if self.monitor_types is not None:
			return [
				(
					monitor_name,
					monitor_name,
					monitor_type + ' Monitor Data',
					'',
					i,
				)
				for i, (monitor_name, monitor_type) in enumerate(
					self.monitor_types.items()
				)
			]

		return []

	####################
	# - Properties: Monitor Information
	####################
	@bl_cache.cached_bl_property(depends_on={'sim_datas', 'monitor_name'})
	def monitor_datas(self) -> SimDataArrayInfo | None:
		"""Extract the currently selected monitor's data from all simulation datas in the batch."""
		if self.sim_datas is not None and self.monitor_name is not None:
			return {
				syms_vals: sim_data.monitor_data.get(self.monitor_name)
				for syms_vals, sim_data in self.sim_datas.items()
			}
		return None

	@bl_cache.cached_bl_property(depends_on={'example_sim_data', 'monitor_name'})
	def valid_monitor_attrs(self) -> SimDataArrayInfo | None:
		"""Valid attributes of the monitor, from the example sim data under the presumption that the entire batch shares the same attribute validity."""
		if self.example_sim_data is not None and self.monitor_name is not None:
			return valid_monitor_attrs(self.example_sim_data, self.monitor_name)
		return None

	####################
	# - UI
	####################
	def draw_label(self) -> None:
		"""Show the extracted data (if any) in the node's header label.

		Notes:
			Called by Blender to determine the text to place in the node's header.
		"""
		if self.monitor_name is not None:
			return f'Extract: {self.monitor_name}'

		return self.bl_label

	def draw_props(self, _: bpy.types.Context, col: bpy.types.UILayout) -> None:
		"""Draw node properties in the node.

		Parameters:
			col: UI target for drawing.
		"""
		col.prop(self, self.blfields['monitor_name'], text='')

	####################
	# - FlowKind.Func
	####################
	@events.computes_output_socket(
		'Expr',
		kind=FK.Func,
		# Loaded
		props={'monitor_datas', 'valid_monitor_attrs'},
	)
	def compute_extracted_data_func(self, props) -> ct.FuncFlow | FS:
		"""Aggregates the selected monitor's data across all batched symbolic realizations, into a single FuncFlow."""
		monitor_datas = props['monitor_datas']
		valid_monitor_attrs = props['valid_monitor_attrs']

		if monitor_datas is not None and valid_monitor_attrs is not None:
			monitor_datas_xarrs = extract_monitor_xarrs(
				monitor_datas, valid_monitor_attrs
			)

			example_monitor_data = next(iter(monitor_datas.values()))
			monitor_type = example_monitor_data.type.removesuffix('Data')
			output_sym = output_symbol_by_type(monitor_type)

			# Stack Inner Dimensions: components | *
			## -> Each realization maps to exactly one xarray.
			## -> We extract its data, and wrap it into a FuncFlow.
			## -> This represents the "inner" dimensions, with components|data.
			## -> We then attach this singular FuncFlow to that realization.
			inner_funcs = {}
			for syms_vals, attr_xarrs in monitor_datas_xarrs.items():
				# XArray Capture Function
				## -> We can't generally capture a loop variable inline.
				## -> By making a new function, we get a new scope.
				def _xarr_values(xarr):
					return lambda: xarr.values

				# Bind XArray Values into FuncFlows
				## -> Each monitor component has an xarray.
				funcs = [
					ct.FuncFlow(
						func=_xarr_values(xarr),
						func_output=output_sym,
						supports_jax=True,
					)
					for xarr in attr_xarrs.values()
				]
				log.critical(['FUNCS', funcs])
				# Single Component: No Stacking of Dimensions - *
				if len(funcs) == 1:
					inner_funcs[syms_vals] = funcs[0]

				# Many Components: Stack Dimensions - components | *
				else:
					inner_funcs[syms_vals] = functools.reduce(
						lambda a, b: a | b, funcs
					).compose_within(
						lambda els: jnp.stack(els, axis=0),
						enclosing_func_output=output_sym,
					)

			# Stack Batch Dims: vals0 | vals1 | ... | valsN | components | *
			## -> We stack the inner-dimensional object together, backwards.
			## -> Each stack prepends a new dimension.
			## -> Here, everything is integer-indexed.
			## -> But in the InfoFlow, a similar process contextualizes idxs.
			example_syms_vals = next(iter(monitor_datas.keys()))
			syms = example_syms_vals[0] if example_syms_vals != () else ()
			outer_funcs = inner_funcs
			log.critical(['INNER FUNCS', inner_funcs])
			for _, axis in reversed(list(enumerate(syms))):
				log.critical([axis, outer_funcs])
				new_outer_funcs = {}
				# Collect Funcs Along *vals[axis] | ...
				## -> Grab ONLY up to 'axis' syms_vals.
				## -> '|' all functions that share axis-deficient syms_vals.
				## -> Thus, we '|' functions along the last axis.
				for unreduced_syms_vals, func in outer_funcs:
					reduced_syms_vals = unreduced_syms_vals[:axis]
					if reduced_syms_vals in new_outer_funcs:
						new_outer_funcs[reduced_syms_vals] = (
							new_outer_funcs[reduced_syms_vals] | func
						)
					else:
						new_outer_funcs[reduced_syms_vals] = func

				# Aggregate All Collected Funcs
				## -> Any functions that went through a | are stacked.
				## -> Otherwise, just add a len=1 dimension.
				new_reduced_outer_funcs = {
					reduced_syms_vals: (
						combined_func.compose_within(
							lambda els: jnp.stack(els, axis=0),
							enclosing_func_output=output_sym,
						)
						if combined_func.is_concatenated  ## Went through a |
						else combined_func.compose_within(
							lambda el: jnp.expand_dims(el, axis=0),
							enclosing_func_output=output_sym,
						)
					)
					for reduced_syms_vals, combined_func in new_outer_funcs.items()
				}

				# Reset Outer Funcs to Axis-Deficient Reduction
				## -> This effectively removes + aggregates the last axis.
				## -> When the loop is done, only {(): val} will be left.
				outer_funcs = new_reduced_outer_funcs
			return next(iter(outer_funcs.values()))
		return FS.FlowPending

	####################
	# - FlowKind.Info
	####################
	@events.computes_output_socket(
		'Expr',
		kind=FK.Info,
		# Loaded
		props={'monitor_datas', 'valid_monitor_attrs'},
	)
	def compute_extracted_data_info(self, props) -> ct.InfoFlow | FS:
		"""Declare `Data:Info` by manually selecting appropriate axes, units, etc. for each monitor type.

		Returns:
			Information describing the `Data:Func`, if available, else `FS.FlowPending`.
		"""
		monitor_datas = props['monitor_datas']
		valid_monitor_attrs = props['valid_monitor_attrs']

		if monitor_datas is not None and valid_monitor_attrs is not None:
			return extract_info(monitor_datas, valid_monitor_attrs)
		return FS.FlowPending

	####################
	# - FlowKind.Params
	####################
	@events.computes_output_socket(
		'Expr',
		kind=FK.Params,
	)
	def compute_params(self) -> ct.ParamsFlow:
		"""Declare an empty `Data:Params`, to indicate the start of a function-composition pipeline.

		Returns:
			A completely empty `ParamsFlow`, ready to be composed.
		"""
		return ct.ParamsFlow()

	####################
	# - Log: FlowKind.Value|Array
	####################
	@events.computes_output_socket(
		'Log',
		kind=FK.Value,
		# Loaded
		props={'sim_datas'},
	)
	def compute_extracted_log(self, props) -> str | FS:
		"""Extract the log from a single simulation that ran."""
		sim_datas = props['sim_datas']
		if sim_datas is not None and len(sim_datas) == 1:
			sim_data = next(iter(sim_datas.values()))
			if sim_data.log is not None:
				return sim_data.log
		return FS.FlowPending

	@events.computes_output_socket(
		'Log',
		kind=FK.Array,
		# Loaded
		props={'sim_datas'},
	)
	def compute_extracted_logs(self, props) -> dict[RealizedSymsVals, str] | FS:
		"""Extract the log from all simulation that ran in the batch."""
		sim_datas = props['sim_datas']
		if sim_datas is not None and sim_datas:
			return {
				syms_vals: sim_data.log if sim_data.log is not None else ''
				for syms_vals, sim_data in sim_datas.items()
			}
		return FS.FlowPending


####################
# - Blender Registration
####################
BL_REGISTER = [
	ExtractDataNode,
]
BL_NODES = {ct.NodeType.ExtractData: (ct.NodeCategory.MAXWELLSIM_ANALYSIS)}
