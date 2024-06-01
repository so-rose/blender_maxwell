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

from blender_maxwell.utils import bl_cache, logger, sim_symbols
from blender_maxwell.utils import sympy_extra as spux

from ... import contracts as ct
from ... import sockets
from .. import base, events

log = logger.get(__name__)

TDMonitorData: typ.TypeAlias = td.components.data.monitor_data.MonitorData


####################
# - Monitor Label Arrays
####################
def valid_monitor_attrs(sim_data: td.SimulationData, monitor_name: str) -> list[str]:
	"""Retrieve the valid attributes of `sim_data.monitor_data' from a valid `sim_data` of type `td.SimulationData`.

	Parameters:
		monitor_type: The name of the monitor type, with the 'Data' prefix removed.
	"""
	monitor_data = sim_data.monitor_data[monitor_name]
	monitor_type = monitor_data.type

	match monitor_type:
		case 'Field' | 'FieldTime' | 'Mode':
			## TODO: flux, poynting, intensity
			return [
				field_component
				for field_component in ['Ex', 'Ey', 'Ez', 'Hx', 'Hy', 'Hz']
				if getattr(monitor_data, field_component, None) is not None
			]

		case 'Permittivity':
			return ['eps_xx', 'eps_yy', 'eps_zz']

		case 'Flux' | 'FluxTime':
			return ['flux']

		case (
			'FieldProjectionAngle'
			| 'FieldProjectionCartesian'
			| 'FieldProjectionKSpace'
			| 'Diffraction'
		):
			return [
				'Er',
				'Etheta',
				'Ephi',
				'Hr',
				'Htheta',
				'Hphi',
			]


def extract_info(monitor_data, monitor_attr: str) -> ct.InfoFlow | None:  # noqa: PLR0911
	"""Extract an InfoFlow encapsulating raw data contained in an attribute of the given monitor data."""
	xarr = getattr(monitor_data, monitor_attr, None)
	if xarr is None:
		return None

	def mk_idx_array(axis: str) -> ct.RangeFlow | ct.ArrayFlow:
		return ct.RangeFlow.try_from_array(
			ct.ArrayFlow(
				values=xarr.get_index(axis).values,
				unit=symbols[axis].unit,
				is_sorted=True,
			)
		)

	# Compute InfoFlow from XArray
	symbols = {
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
		# Cartesian Fields
		'Ex': sim_symbols.field_ex(spu.volt / spu.micrometer),
		'Ey': sim_symbols.field_ey(spu.volt / spu.micrometer),
		'Ez': sim_symbols.field_ez(spu.volt / spu.micrometer),
		'Hx': sim_symbols.field_hx(spu.volt / spu.micrometer),
		'Hy': sim_symbols.field_hy(spu.volt / spu.micrometer),
		'Hz': sim_symbols.field_hz(spu.volt / spu.micrometer),
		# Spherical Fields
		'Er': sim_symbols.field_er(spu.volt / spu.micrometer),
		'Etheta': sim_symbols.ang_theta(spu.volt / spu.micrometer),
		'Ephi': sim_symbols.field_ez(spu.volt / spu.micrometer),
		'Hr': sim_symbols.field_hr(spu.volt / spu.micrometer),
		'Htheta': sim_symbols.field_hy(spu.volt / spu.micrometer),
		'Hphi': sim_symbols.field_hz(spu.volt / spu.micrometer),
		# Wavevector
		'ux': sim_symbols.dir_x(spu.watt),
		'uy': sim_symbols.dir_y(spu.watt),
		# Diffraction Orders
		'orders_x': sim_symbols.diff_order_x(None),
		'orders_y': sim_symbols.diff_order_y(None),
	}

	match monitor_data.type:
		case 'Field' | 'FieldProjectionCartesian' | 'Permittivity' | 'Mode':
			return ct.InfoFlow(
				dims={
					symbols['x']: mk_idx_array('x'),
					symbols['y']: mk_idx_array('y'),
					symbols['z']: mk_idx_array('z'),
					symbols['f']: mk_idx_array('f'),
				},
				output=symbols[monitor_attr],
			)

		case 'FieldTime':
			return ct.InfoFlow(
				dims={
					symbols['x']: mk_idx_array('x'),
					symbols['y']: mk_idx_array('y'),
					symbols['z']: mk_idx_array('z'),
					symbols['t']: mk_idx_array('t'),
				},
				output=symbols[monitor_attr],
			)

		case 'Flux':
			return ct.InfoFlow(
				dims={
					symbols['f']: mk_idx_array('f'),
				},
				output=symbols[monitor_attr],
			)

		case 'FluxTime':
			return ct.InfoFlow(
				dims={
					symbols['t']: mk_idx_array('t'),
				},
				output=symbols[monitor_attr],
			)

		case 'FieldProjectionAngle':
			return ct.InfoFlow(
				dims={
					symbols['r']: mk_idx_array('r'),
					symbols['theta']: mk_idx_array('theta'),
					symbols['phi']: mk_idx_array('phi'),
					symbols['f']: mk_idx_array('f'),
				},
				output=symbols[monitor_attr],
			)

		case 'FieldProjectionKSpace':
			return ct.InfoFlow(
				dims={
					symbols['ux']: mk_idx_array('ux'),
					symbols['uy']: mk_idx_array('uy'),
					symbols['r']: mk_idx_array('r'),
					symbols['f']: mk_idx_array('f'),
				},
				output=symbols[monitor_attr],
			)

		case 'Diffraction':
			return ct.InfoFlow(
				dims={
					symbols['orders_x']: mk_idx_array('orders_x'),
					symbols['orders_y']: mk_idx_array('orders_y'),
					symbols['f']: mk_idx_array('f'),
				},
				output=symbols[monitor_attr],
			)

	return None


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
		'Sim Data': sockets.MaxwellFDTDSimDataSocketDef(),
	}
	output_socket_sets: typ.ClassVar = {
		'Expr': sockets.ExprSocketDef(active_kind=ct.FlowKind.Func),
	}

	####################
	# - Properties: Monitor Name
	####################
	@events.on_value_changed(
		socket_name='Sim Data',
		input_sockets={'Sim Data'},
		input_sockets_optional={'Sim Data': True},
	)
	def on_sim_data_changed(self, input_sockets) -> None:  # noqa: D102
		has_sim_data = not ct.FlowSignal.check(input_sockets['Sim Data'])
		if has_sim_data:
			self.sim_data = bl_cache.Signal.InvalidateCache

	@bl_cache.cached_bl_property()
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

	@bl_cache.cached_bl_property(depends_on={'sim_data'})
	def sim_data_monitor_nametype(self) -> dict[str, str] | None:
		"""Dictionary from monitor names on `self.sim_data` to their associated type name (with suffix 'Data' removed).

		Return:
			The name to type of monitors in the simulation data.
		"""
		if self.sim_data is not None:
			return {
				monitor_name: monitor_data.type.removesuffix('Data')
				for monitor_name, monitor_data in self.sim_data.monitor_data.items()
			}

		return None

	monitor_name: enum.StrEnum = bl_cache.BLField(
		enum_cb=lambda self, _: self.search_monitor_names(),
		cb_depends_on={'sim_data_monitor_nametype'},
	)

	def search_monitor_names(self) -> list[ct.BLEnumElement]:
		"""Compute valid values for `self.monitor_attr`, for a dynamic `EnumProperty`.

		Notes:
			Should be reset (via `self.monitor_attr`) with (after) `self.sim_data_monitor_nametype`, `self.monitor_data_attrs`, and (implicitly) `self.monitor_type`.

			See `bl_cache.BLField` for more on dynamic `EnumProperty`.

		Returns:
			Valid `self.monitor_attr` in a format compatible with dynamic `EnumProperty`.
		"""
		if self.sim_data_monitor_nametype is not None:
			return [
				(
					monitor_name,
					monitor_name,
					monitor_type + ' Monitor Data',
					'',
					i,
				)
				for i, (monitor_name, monitor_type) in enumerate(
					self.sim_data_monitor_nametype.items()
				)
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

		if has_sim_data:
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
		kind=ct.FlowKind.Func,
		# Loaded
		props={'monitor_name'},
		input_sockets={'Sim Data'},
		input_socket_kinds={'Sim Data': ct.FlowKind.Value},
	)
	def compute_expr(
		self, props: dict, input_sockets: dict
	) -> ct.FuncFlow | ct.FlowSignal:
		sim_data = input_sockets['Sim Data']
		monitor_name = props['monitor_name']

		has_sim_data = not ct.FlowSignal.check(sim_data)

		if has_sim_data and monitor_name is not None:
			monitor_data = sim_data.get(monitor_name)
			if monitor_data is not None:
				# Extract Valid Index Labels
				## -> The first output axis will be integer-indexed.
				## -> Each integer will have a string label.
				## -> Those string labels explain the integer as ex. Ex, Ey, Hy.
				idx_labels = valid_monitor_attrs(sim_data, monitor_name)

				# Extract Info
				## -> We only need the output symbol.
				## -> All labelled outputs have the same output SimSymbol.
				info = extract_info(monitor_data, idx_labels[0])

				# Generate FuncFlow Per Index Label
				## -> We extract each XArray as an attribute of monitor_data.
				## -> We then bind its values into a unique func_flow.
				## -> This lets us 'stack' then all along the first axis.
				func_flows = []
				for idx_label in idx_labels:
					xarr = getattr(monitor_data, idx_label)
					func_flows.append(
						ct.FuncFlow(
							func=lambda xarr=xarr: xarr.values,
							supports_jax=True,
						)
					)

				# Concatenate and Stack Unified FuncFlow
				## -> First, 'reduce' lets us __or__ all the FuncFlows together.
				## -> Then, 'compose_within' lets us stack them along axis=0.
				## -> The "new" axis=0 is int-indexed axis w/idx_labels labels!
				return functools.reduce(lambda a, b: a | b, func_flows).compose_within(
					lambda data: jnp.stack(data, axis=0),
					func_output=info.output,
				)
			return ct.FlowSignal.FlowPending
		return ct.FlowSignal.FlowPending

	####################
	# - FlowKind.Params
	####################
	@events.computes_output_socket(
		'Expr',
		kind=ct.FlowKind.Params,
		input_sockets={'Sim Data'},
		input_socket_kinds={'Sim Data': ct.FlowKind.Params},
	)
	def compute_data_params(self, input_sockets) -> ct.ParamsFlow:
		"""Declare an empty `Data:Params`, to indicate the start of a function-composition pipeline.

		Returns:
			A completely empty `ParamsFlow`, ready to be composed.
		"""
		sim_params = input_sockets['Sim Data']
		has_sim_params = not ct.FlowSignal.check(sim_params)

		if has_sim_params:
			return sim_params
		return ct.ParamsFlow()

	####################
	# - FlowKind.Info
	####################
	@events.computes_output_socket(
		'Expr',
		kind=ct.FlowKind.Info,
		# Loaded
		props={'monitor_name'},
		input_sockets={'Sim Data'},
		input_socket_kinds={'Sim Data': ct.FlowKind.Value},
	)
	def compute_extracted_data_info(self, props, input_sockets) -> ct.InfoFlow:
		"""Declare `Data:Info` by manually selecting appropriate axes, units, etc. for each monitor type.

		Returns:
			Information describing the `Data:Func`, if available, else `ct.FlowSignal.FlowPending`.
		"""
		sim_data = input_sockets['Sim Data']
		monitor_name = props['monitor_name']

		has_sim_data = not ct.FlowSignal.check(sim_data)

		if not has_sim_data or monitor_name is None:
			return ct.FlowSignal.FlowPending

		# Extract Data
		## -> All monitor_data.<idx_label> have the exact same InfoFlow.
		## -> So, just construct an InfoFlow w/prepended labelled dimension.
		monitor_data = sim_data.get(monitor_name)
		idx_labels = valid_monitor_attrs(sim_data, monitor_name)
		info = extract_info(monitor_data, idx_labels[0])

		return info.prepend_dim(sim_symbols.idx, idx_labels)


####################
# - Blender Registration
####################
BL_REGISTER = [
	ExtractDataNode,
]
BL_NODES = {ct.NodeType.ExtractData: (ct.NodeCategory.MAXWELLSIM_ANALYSIS)}
