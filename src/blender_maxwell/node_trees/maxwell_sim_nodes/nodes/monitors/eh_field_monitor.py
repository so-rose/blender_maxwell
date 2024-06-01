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
			physical_type=spux.PhysicalType.Length,
		),
		'Size': sockets.ExprSocketDef(
			size=spux.NumberSize1D.Vec3,
			physical_type=spux.PhysicalType.Length,
			default_value=sp.Matrix([1, 1, 1]),
			abs_min=0,
		),
		'Stride': sockets.ExprSocketDef(
			size=spux.NumberSize1D.Vec3,
			mathtype=spux.MathType.Integer,
			default_value=sp.Matrix([10, 10, 10]),
			abs_min=0,
		),
	}
	input_socket_sets: typ.ClassVar = {
		'Freq Domain': {
			'Freqs': sockets.ExprSocketDef(
				active_kind=ct.FlowKind.Range,
				physical_type=spux.PhysicalType.Freq,
				default_unit=spux.THz,
				default_min=374.7406,  ## 800nm
				default_max=1498.962,  ## 200nm
				default_steps=100,
			),
		},
		'Time Domain': {
			't Range': sockets.ExprSocketDef(
				active_kind=ct.FlowKind.Range,
				physical_type=spux.PhysicalType.Time,
				default_unit=spu.picosecond,
				default_min=0,
				default_max=10,
				default_steps=0,
			),
			't Stride': sockets.ExprSocketDef(
				mathtype=spux.MathType.Integer,
				default_value=100,
			),
		},
	}
	output_sockets: typ.ClassVar = {
		'Monitor': sockets.MaxwellMonitorSocketDef(active_kind=ct.FlowKind.Func),
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
		layout.prop(self, self.blfields['fields'], expand=True)

	####################
	# - FlowKind.Value
	####################
	@events.computes_output_socket(
		'Monitor',
		kind=ct.FlowKind.Value,
		# Loaded
		output_sockets={'Monitor'},
		output_socket_kinds={'Monitor': {ct.FlowKind.Func, ct.FlowKind.Params}},
	)
	def compute_value(self, output_sockets) -> ct.ParamsFlow | ct.FlowSignal:
		"""Compute the particular value of the simulation domain from strictly non-symbolic inputs."""
		output_func = output_sockets['Monitor'][ct.FlowKind.Func]
		output_params = output_sockets['Monitor'][ct.FlowKind.Params]

		has_output_func = not ct.FlowSignal.check(output_func)
		has_output_params = not ct.FlowSignal.check(output_params)

		if has_output_func and has_output_params and not output_params.symbols:
			return output_func.realize(output_params, disallow_jax=True)
		return ct.FlowSignal.FlowPending

	####################
	# - FlowKind.Func
	####################
	@events.computes_output_socket(
		'Monitor',
		kind=ct.FlowKind.Func,
		# Loaded
		props={'active_socket_set', 'sim_node_name', 'fields'},
		input_sockets={
			'Center',
			'Size',
			'Stride',
			'Freqs',
			't Range',
			't Stride',
		},
		input_socket_kinds={
			'Center': ct.FlowKind.Func,
			'Size': ct.FlowKind.Func,
			'Stride': ct.FlowKind.Func,
			'Freqs': ct.FlowKind.Func,
			't Range': ct.FlowKind.Func,
			't Stride': ct.FlowKind.Func,
		},
	)
	def compute_func(self, props, input_sockets) -> td.FieldMonitor:
		center = input_sockets['Center']
		size = input_sockets['Size']
		stride = input_sockets['Stride']

		has_center = not ct.FlowSignal.check(center)
		has_size = not ct.FlowSignal.check(size)
		has_stride = not ct.FlowSignal.check(stride)

		if has_center and has_size and has_stride:
			name = props['sim_node_name']
			fields = props['fields']

			common_func_flow = (
				center.scale_to_unit_system(ct.UNITS_TIDY3D)
				| size.scale_to_unit_system(ct.UNITS_TIDY3D)
				| stride
			)

			match props['active_socket_set']:
				case 'Freq Domain':
					freqs = input_sockets['Freqs']
					has_freqs = not ct.FlowSignal.check(freqs)

					if has_freqs:
						return (
							common_func_flow
							| freqs.scale_to_unit_system(ct.UNITS_TIDY3D)
						).compose_within(
							lambda els: td.FieldMonitor(
								center=els[0].flatten().tolist(),
								size=els[1].flatten().tolist(),
								name=name,
								interval_space=els[2].flatten().tolist(),
								freqs=els[3].flatten(),
								fields=fields,
							)
						)

				case 'Time Domain':
					t_range = input_sockets['t Range']
					t_stride = input_sockets['t Stride']

					has_t_range = not ct.FlowSignal.check(t_range)
					has_t_stride = not ct.FlowSignal.check(t_stride)

					if has_t_range and has_t_stride:
						return (
							common_func_flow
							| t_range.scale_to_unit_system(ct.UNITS_TIDY3D)
							| t_stride.scale_to_unit_system(ct.UNITS_TIDY3D)
						).compose_within(
							lambda els: td.FieldTimeMonitor(
								center=els[0].flatten().tolist(),
								size=els[1].flatten().tolist(),
								name=name,
								interval_space=els[2].flatten().tolist(),
								start=els[3][0],
								stop=els[3][-1],
								interval=els[4],
								fields=fields,
							)
						)
		return ct.FlowSignal.FlowPending

	####################
	# - FlowKind.Params
	####################
	@events.computes_output_socket(
		'Monitor',
		kind=ct.FlowKind.Params,
		# Loaded
		props={'active_socket_set'},
		input_sockets={
			'Center',
			'Size',
			'Stride',
			'Freqs',
			't Range',
			't Stride',
		},
		input_socket_kinds={
			'Center': ct.FlowKind.Params,
			'Size': ct.FlowKind.Params,
			'Stride': ct.FlowKind.Params,
			'Freqs': ct.FlowKind.Params,
			't Range': ct.FlowKind.Params,
			't Stride': ct.FlowKind.Params,
		},
	)
	def compute_params(self, props, input_sockets) -> None:
		center = input_sockets['Center']
		size = input_sockets['Size']
		stride = input_sockets['Stride']

		has_center = not ct.FlowSignal.check(center)
		has_size = not ct.FlowSignal.check(size)
		has_stride = not ct.FlowSignal.check(stride)

		if has_center and has_size and has_stride:
			common_params = center | size | stride
			match props['active_socket_set']:
				case 'Freq Domain':
					freqs = input_sockets['Freqs']
					has_freqs = not ct.FlowSignal.check(freqs)

					if has_freqs:
						return common_params | freqs

				case 'Time Domain':
					t_range = input_sockets['t Range']
					t_stride = input_sockets['t Stride']

					has_t_range = not ct.FlowSignal.check(t_range)
					has_t_stride = not ct.FlowSignal.check(t_stride)

					if has_t_range and has_t_stride:
						return common_params | t_range | t_stride
		return ct.FlowSignal.FlowPending

	####################
	# - Preview
	####################
	@events.computes_output_socket(
		'Monitor',
		kind=ct.FlowKind.Previews,
		# Loaded
		props={'sim_node_name'},
		output_sockets={'Monitor'},
		output_socket_kinds={'Monitor': ct.FlowKind.Params},
	)
	def compute_previews(self, props, output_sockets):
		output_params = output_sockets['Monitor']
		has_output_params = not ct.FlowSignal.check(output_params)

		if has_output_params and not output_params.symbols:
			return ct.PreviewsFlow(bl_object_names={props['sim_node_name']})
		return ct.PreviewsFlow()

	@events.on_value_changed(
		# Trigger
		socket_name={'Center', 'Size'},
		run_on_init=True,
		# Loaded
		managed_objs={'modifier'},
		input_sockets={'Center', 'Size'},
		output_sockets={'Monitor'},
		output_socket_kinds={'Monitor': ct.FlowKind.Params},
	)
	def on_previewable_changed(self, managed_objs, input_sockets, output_sockets):
		center = input_sockets['Center']
		size = input_sockets['Size']
		output_params = output_sockets['Monitor']

		has_center = not ct.FlowSignal.check(center)
		has_size = not ct.FlowSignal.check(size)
		has_output_params = not ct.FlowSignal.check(output_params)

		if has_center and has_size and has_output_params and not output_params.symbols:
			# Push Input Values to GeoNodes Modifier
			managed_objs['modifier'].bl_modifier(
				'NODES',
				{
					'node_group': import_geonodes(GeoNodes.MonitorEHField),
					'unit_system': ct.UNITS_BLENDER,
					'inputs': {
						'Size': size,
					},
				},
				location=spux.scale_to_unit_system(center, ct.UNITS_BLENDER),
			)


####################
# - Blender Registration
####################
BL_REGISTER = [
	EHFieldMonitorNode,
]
BL_NODES = {ct.NodeType.EHFieldMonitor: (ct.NodeCategory.MAXWELLSIM_MONITORS)}
