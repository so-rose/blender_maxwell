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

"""Implements `PowerFluxMonitorNode`."""

import itertools
import typing as typ

import bpy
import sympy as sp
import sympy.physics.units as spu
import tidy3d as td

from blender_maxwell.assets.geonodes import GeoNodes, import_geonodes
from blender_maxwell.utils import bl_cache, logger
from blender_maxwell.utils import sympy_extra as spux

from ... import contracts as ct
from ... import managed_objs, sockets
from .. import base, events

log = logger.get(__name__)
FK = ct.FlowKind
FS = ct.FlowSignal

ALL_3D_DIRS = {
	ax + sgn for ax, sgn in itertools.product(set(ct.SimSpaceAxis), set(ct.SimAxisDir))
}


class PowerFluxMonitorNode(base.MaxwellSimNode):
	"""Node providing for the monitoring of electromagnetic field flux a given planar region or volume, in either the frequency or the time domain."""

	node_type = ct.NodeType.PowerFluxMonitor
	bl_label = 'Power Flux Monitor'
	use_sim_node_name = True

	####################
	# - Sockets
	####################
	input_sockets: typ.ClassVar = {
		'Center': sockets.ExprSocketDef(
			size=spux.NumberSize1D.Vec3,
			physical_type=spux.PhysicalType.Length,
		),
		'Size': sockets.ExprSocketDef(
			size=spux.NumberSize1D.Vec3,
			physical_type=spux.PhysicalType.Length,
			default_value=sp.ImmutableMatrix([1, 1, 1]),
			abs_min=0,
			abs_min_closed=False,
		),
		'Stride': sockets.ExprSocketDef(
			size=spux.NumberSize1D.Vec3,
			mathtype=spux.MathType.Integer,
			default_value=sp.ImmutableMatrix([10, 10, 10]),
			abs_min=1,
		),
	}
	input_socket_sets: typ.ClassVar = {
		'Freq Domain': {
			'Freqs': sockets.ExprSocketDef(
				active_kind=FK.Range,
				physical_type=spux.PhysicalType.Freq,
				default_unit=spux.THz,
				default_min=374.7406,  ## 800nm
				default_max=1498.962,  ## 200nm
				default_steps=100,
			),
		},
		'Time Domain': {
			't Range': sockets.ExprSocketDef(
				active_kind=FK.Range,
				physical_type=spux.PhysicalType.Time,
				default_unit=spu.picosecond,
				default_min=0,
				default_max=10,
				default_steps=2,
			),
			't Stride': sockets.ExprSocketDef(
				mathtype=spux.MathType.Integer,
				default_value=100,
				abs_min=1,
			),
		},
	}
	output_sockets: typ.ClassVar = {
		'Monitor': sockets.MaxwellMonitorSocketDef(),
	}

	managed_obj_types: typ.ClassVar = {
		'modifier': managed_objs.ManagedBLModifier,
	}

	####################
	# - Properties
	####################
	direction_2d: ct.SimAxisDir = bl_cache.BLField(ct.SimAxisDir.Plus)
	include_3d: set[ct.SimSpaceAxis] = bl_cache.BLField(set(ct.SimSpaceAxis))
	include_3d_x: set[ct.SimAxisDir] = bl_cache.BLField(set(ct.SimAxisDir))
	include_3d_y: set[ct.SimAxisDir] = bl_cache.BLField(set(ct.SimAxisDir))
	include_3d_z: set[ct.SimAxisDir] = bl_cache.BLField(set(ct.SimAxisDir))

	####################
	# - UI
	####################
	def draw_props(self, _: bpy.types.Context, layout: bpy.types.UILayout) -> None:
		"""Draw the properties of the node."""
		# 2D Monitor
		if 0 in self._compute_input('Size'):
			layout.prop(self, self.blfields['direction_2d'], expand=True)

		# 3D Monitor
		else:
			layout.prop(self, self.blfields['include_3d'], expand=True)
			row = layout.row(align=False)
			if ct.SimSpaceAxis.X in self.include_3d:
				row.prop(self, self.blfields['include_3d_x'], expand=True)
			if ct.SimSpaceAxis.Y in self.include_3d:
				row.prop(self, self.blfields['include_3d_y'], expand=True)
			if ct.SimSpaceAxis.Z in self.include_3d:
				row.prop(self, self.blfields['include_3d_z'], expand=True)

	####################
	# - FlowKind.Value
	####################
	@events.computes_output_socket(
		'Monitor',
		kind=FK.Value,
		# Loaded
		output_sockets={'Monitor'},
		output_socket_kinds={
			'Monitor': {FK.Func, FK.Params},
		},
	)
	def compute_value(self, output_sockets) -> ct.ParamsFlow | FS:
		"""Realizes the output function w/output parameters."""
		value = events.realize_known(output_sockets['Monitor'])
		if value is not None:
			return value
		return FS.FlowPending

	####################
	# - FlowKind.Func
	####################
	@events.computes_output_socket(
		'Monitor',
		kind=FK.Func,
		# Loaded
		props={'active_socket_set', 'sim_node_name', 'direction_2d'},
		inscks_kinds={
			'Center': FK.Func,
			'Size': FK.Func,
			'Stride': FK.Func,
			'Freqs': FK.Func,
			't Range': FK.Func,
			't Stride': FK.Func,
		},
		input_sockets_optional={'Freqs', 't Range', 't Stride'},
		scale_input_sockets={
			'Center': ct.UNITS_TIDY3D,
			'Size': ct.UNITS_TIDY3D,
			'Freqs': ct.UNITS_TIDY3D,
			't Range': ct.UNITS_TIDY3D,
		},
	)
	def compute_func(self, input_sockets, props) -> ct.FuncFlow:  # noqa: C901
		"""Compute the correct flux monitor."""
		center = input_sockets['Center']
		size = input_sockets['Size']
		stride = input_sockets['Stride']
		freqs = input_sockets['Freqs']
		t_range = input_sockets['t Range']
		t_stride = input_sockets['t Stride']

		sim_node_name = props['sim_node_name']
		direction_2d = props['direction_2d']

		# 3D Flux Monitor: Computed Excluded Directions
		## -> The flux is always recorded outgoing.
		## -> However, one can exclude certain faces from participating.
		include_3d = props['include_3d']
		include_3d_x = props['include_3d_x']
		include_3d_y = props['include_3d_y']
		include_3d_z = props['include_3d_z']
		excluded_3d = set()
		if ct.SimSpaceAxis.X in include_3d:
			if ct.SimAxisDir.Plus in include_3d_x:
				excluded_3d.add('x+')
			if ct.SimAxisDir.Minus in include_3d_x:
				excluded_3d.add('x-')

		if ct.SimSpaceAxis.Y in include_3d:
			if ct.SimAxisDir.Plus in include_3d_y:
				excluded_3d.add('y+')
			if ct.SimAxisDir.Minus in include_3d_y:
				excluded_3d.add('y-')

		if ct.SimSpaceAxis.Z in include_3d:
			if ct.SimAxisDir.Plus in include_3d_z:
				excluded_3d.add('z+')
			if ct.SimAxisDir.Minus in include_3d_z:
				excluded_3d.add('z-')

		excluded_3d = tuple(ALL_3D_DIRS - excluded_3d)

		# Compute Monitor
		common_func = center | size | stride
		active_socket_set = props['active_socket_set']
		match active_socket_set:
			case 'Freq Domain' if not FS.check(freqs):
				return (common_func | freqs).compose_within(
					lambda els: td.FluxMonitor(
						name=sim_node_name,
						center=els[0],
						size=els[1],
						interval_space=els[2],
						freqs=els[3],
						normal_dir=direction_2d.plus_or_minus,
						exclude_surfaces=excluded_3d,
					)
				)

			case 'Time Domain' if not FS.check(t_range) and not FS.check(t_stride):
				return (common_func | t_range | t_stride).compose_within(
					lambda els: td.FluxTimeMonitor(
						name=sim_node_name,
						center=els[0],
						size=els[1],
						interval_space=els[2],
						start=els[3].item(0),
						stop=els[3].item(1),
						interval=els[4],
						normal_dir=direction_2d.plus_or_minus,
						exclude_surfaces=excluded_3d,
					)
				)

		return FS.FlowPending

	####################
	# - FlowKind.Params
	####################
	@events.computes_output_socket(
		'Monitor',
		kind=FK.Params,
		# Loaded
		props={'active_socket_set'},
		inscks_kinds={
			'Center': FK.Params,
			'Size': FK.Params,
			'Stride': FK.Params,
			'Freqs': FK.Params,
			't Range': FK.Params,
			't Stride': FK.Params,
		},
		input_sockets_optional={'Freqs', 't Range', 't Stride'},
	)
	def compute_params(self, input_sockets, props) -> ct.ParamsFlow:
		"""Compute the function parameters of the monitor."""
		center = input_sockets['Center']
		size = input_sockets['Size']
		stride = input_sockets['Stride']

		freqs = input_sockets['Freqs']
		t_range = input_sockets['t Range']
		t_stride = input_sockets['t Stride']

		common_params = center | size | stride

		# Compute Monitor
		active_socket_set = props['active_socket_set']
		match active_socket_set:
			case 'Freq Domain' if not FS.check(freqs):
				return common_params | freqs

			case 'Time Domain' if not FS.check(t_range) and not FS.check(t_stride):
				return common_params | t_range | t_stride

		return FS.FlowPending

	####################
	# - Preview - Changes to Input Sockets
	####################
	@events.computes_output_socket(
		'Monitor',
		kind=FK.Previews,
		# Loaded
		props={'sim_node_name'},
	)
	def compute_previews_time(self, props):
		"""Mark the box structure as participating in the preview."""
		return ct.PreviewsFlow(bl_object_names={props['sim_node_name']})

	@events.on_value_changed(
		# Trigger
		socket_name={'Center', 'Size'},
		prop_name={'direction_2d'},
		run_on_init=True,
		# Loaded
		managed_objs={'modifier'},
		inscks_kinds={
			'Center': {FK.Func, FK.Params},
			'Size': {FK.Func, FK.Params},
		},
		scale_input_sockets={
			'Center': ct.UNITS_BLENDER,
			'Size': ct.UNITS_BLENDER,
		},
	)
	def on_previewable_changed(self, managed_objs, input_sockets):
		"""Push changes in the inputs to the center / size."""
		center = events.realize_preview(input_sockets['Center'])
		size = events.realize_preview(input_sockets['Size'])

		managed_objs['modifier'].bl_modifier(
			'NODES',
			{
				'node_group': import_geonodes(GeoNodes.MonitorPowerFlux),
				'inputs': {
					'Size': size,
				},
			},
			location=center,
		)


####################
# - Blender Registration
####################
BL_REGISTER = [
	PowerFluxMonitorNode,
]
BL_NODES = {ct.NodeType.PowerFluxMonitor: (ct.NodeCategory.MAXWELLSIM_MONITORS)}
