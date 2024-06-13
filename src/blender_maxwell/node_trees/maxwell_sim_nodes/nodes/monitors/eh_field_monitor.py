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
MT = spux.MathType
PT = spux.PhysicalType


class EHFieldMonitorNode(base.MaxwellSimNode):
	"""Node providing for the monitoring of electromagnetic fields within a given planar region or volume."""

	node_type = ct.NodeType.EHFieldMonitor
	bl_label = 'EH Field Monitor'
	use_sim_node_name = True

	####################
	# - Sockets
	####################
	input_sockets: typ.ClassVar = {
		'Center': sockets.ExprSocketDef(
			size=spux.NumberSize1D.Vec3,
			physical_type=PT.Length,
		),
		'Size': sockets.ExprSocketDef(
			size=spux.NumberSize1D.Vec3,
			physical_type=PT.Length,
			default_value=sp.ImmutableMatrix([1, 1, 1]),
			abs_min=0,
			abs_min_closed=False,
		),
		'Stride': sockets.ExprSocketDef(
			size=spux.NumberSize1D.Vec3,
			mathtype=MT.Integer,
			default_value=sp.ImmutableMatrix([10, 10, 10]),
			abs_min=1,
		),
	}
	input_socket_sets: typ.ClassVar = {
		'Freq Domain': {
			'Freqs': sockets.ExprSocketDef(
				active_kind=FK.Range,
				physical_type=PT.Freq,
				default_unit=spux.THz,
				default_min=374.7406,  ## 800nm
				default_max=1498.962,  ## 200nm
				default_steps=100,
			),
		},
		'Time Domain': {
			't Range': sockets.ExprSocketDef(
				active_kind=FK.Range,
				physical_type=PT.Time,
				default_unit=spu.picosecond,
				default_min=0,
				default_max=10,
				default_steps=2,
			),
			't Stride': sockets.ExprSocketDef(
				mathtype=MT.Integer,
				default_value=100,
				abs_min=1,
			),
		},
	}
	output_sockets: typ.ClassVar = {
		'Monitor': sockets.MaxwellMonitorSocketDef(active_kind=FK.Func),
	}

	managed_obj_types: typ.ClassVar = {
		'modifier': managed_objs.ManagedBLModifier,
	}

	####################
	# - Properties
	####################
	fields: set[ct.SimFieldPols] = bl_cache.BLField(set(ct.SimFieldPols))

	####################
	# - UI
	####################
	def draw_props(self, _: bpy.types.Context, layout: bpy.types.UILayout) -> None:
		"""Draw the field-selector in the node UI.

		Parameters:
			layout: UI target for drawing.
		"""
		layout.prop(self, self.blfields['fields'], expand=True)

	####################
	# - FlowKind.Value
	####################
	@events.computes_output_socket(
		'Monitor',
		kind=FK.Value,
		# Loaded
		outscks_kinds={
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
		props={'active_socket_set', 'sim_node_name', 'fields'},
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
	def compute_func(self, props, input_sockets) -> td.FieldMonitor:
		"""Lazily assembles the FieldMonitor from the input functions."""
		center = input_sockets['Center']
		size = input_sockets['Size']
		stride = input_sockets['Stride']

		freqs = input_sockets['Freqs']
		t_range = input_sockets['t Range']
		t_stride = input_sockets['t Stride']

		sim_node_name = props['sim_node_name']
		fields = props['fields']

		common_func_flow = center | size | stride
		match props['active_socket_set']:
			case 'Freq Domain' if not FS.check(freqs):
				return (common_func_flow | freqs).compose_within(
					lambda els: td.FieldMonitor(
						name=sim_node_name,
						center=els[0].flatten().tolist(),
						size=els[1].flatten().tolist(),
						interval_space=els[2].flatten().tolist(),
						freqs=els[3].flatten(),
						fields=fields,
					)
				)

			case 'Time Domain' if not FS.check(t_range) and not FS.check(t_stride):
				return (common_func_flow | t_range | t_stride).compose_within(
					lambda els: td.FieldTimeMonitor(
						name=sim_node_name,
						center=els[0].flatten().tolist(),
						size=els[1].flatten().tolist(),
						interval_space=els[2].flatten().tolist(),
						start=els[3][0],
						stop=els[3][-1],
						interval=els[4],
						fields=fields,
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
	def compute_params(self, props, input_sockets) -> None:
		"""Lazily assembles the FieldMonitor from the input functions."""
		center = input_sockets['Center']
		size = input_sockets['Size']
		stride = input_sockets['Stride']

		freqs = input_sockets['Freqs']
		t_range = input_sockets['t Range']
		t_stride = input_sockets['t Stride']

		common_params = center | size | stride
		match props['active_socket_set']:
			case 'Freq Domain' if not FS.check(freqs):
				return common_params | freqs

			case 'Time Domain' if not FS.check(t_range) and not FS.check(t_stride):
				return common_params | t_range | t_stride
		return FS.FlowPending

	####################
	# - Preview
	####################
	@events.computes_output_socket(
		'Monitor',
		kind=FK.Previews,
		# Loaded
		props={'sim_node_name'},
	)
	def compute_previews(self, props):
		"""Mark the monitor as participating in the preview."""
		return ct.PreviewsFlow(bl_object_names={props['sim_node_name']})

	@events.on_value_changed(
		# Trigger
		socket_name={'Center', 'Size'},
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

		# Push Input Values to GeoNodes Modifier
		managed_objs['modifier'].bl_modifier(
			'NODES',
			{
				'node_group': import_geonodes(GeoNodes.MonitorEHField),
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
	EHFieldMonitorNode,
]
BL_NODES = {ct.NodeType.EHFieldMonitor: (ct.NodeCategory.MAXWELLSIM_MONITORS)}
