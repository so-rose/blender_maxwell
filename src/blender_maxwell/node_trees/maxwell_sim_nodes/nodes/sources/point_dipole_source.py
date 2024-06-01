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
import tidy3d as td

from blender_maxwell.assets.geonodes import GeoNodes, import_geonodes
from blender_maxwell.utils import bl_cache, logger
from blender_maxwell.utils import sympy_extra as spux

from ... import contracts as ct
from ... import managed_objs, sockets
from .. import base, events

log = logger.get(__name__)


class PointDipoleSourceNode(base.MaxwellSimNode):
	node_type = ct.NodeType.PointDipoleSource
	bl_label = 'Point Dipole Source'
	use_sim_node_name = True

	####################
	# - Sockets
	####################
	input_sockets: typ.ClassVar = {
		'Temporal Shape': sockets.MaxwellTemporalShapeSocketDef(),
		'Center': sockets.ExprSocketDef(
			size=spux.NumberSize1D.Vec3,
			mathtype=spux.MathType.Real,
			physical_type=spux.PhysicalType.Length,
			default_value=sp.Matrix([0, 0, 0]),
		),
		'Interpolate': sockets.BoolSocketDef(
			default_value=True,
		),
	}
	output_sockets: typ.ClassVar = {
		'Source': sockets.MaxwellSourceSocketDef(active_kind=ct.FlowKind.Func),
	}

	managed_obj_types: typ.ClassVar = {
		'modifier': managed_objs.ManagedBLModifier,
	}

	####################
	# - Properties
	####################
	pol: ct.SimFieldPols = bl_cache.BLField(ct.SimFieldPols.Ex)

	####################
	# - UI
	####################
	def draw_props(self, _: bpy.types.Context, layout: bpy.types.UILayout):
		layout.prop(self, self.blfields['pol'], expand=True)

	####################
	# - FlowKind.Value
	####################
	@events.computes_output_socket(
		'Source',
		kind=ct.FlowKind.Value,
		# Loaded
		output_sockets={'Source'},
		output_socket_kinds={'Source': {ct.FlowKind.Func, ct.FlowKind.Params}},
	)
	def compute_value(self, output_sockets) -> ct.ParamsFlow | ct.FlowSignal:
		"""Compute the particular value of the simulation domain from strictly non-symbolic inputs."""
		output_func = output_sockets['Source'][ct.FlowKind.Func]
		output_params = output_sockets['Source'][ct.FlowKind.Params]

		has_output_func = not ct.FlowSignal.check(output_func)
		has_output_params = not ct.FlowSignal.check(output_params)

		if has_output_func and has_output_params and not output_params.symbols:
			return output_func.realize(output_params, disallow_jax=True)
		return ct.FlowSignal.FlowPending

	####################
	# - FlowKind.Func
	####################
	@events.computes_output_socket(
		'Source',
		kind=ct.FlowKind.Func,
		# Loaded
		props={'pol'},
		input_sockets={'Temporal Shape', 'Center', 'Interpolate'},
		input_socket_kinds={
			'Temporal Shape': ct.FlowKind.Func,
			'Center': ct.FlowKind.Func,
			'Interpolate': ct.FlowKind.Func,
		},
	)
	def compute_func(self, props, input_sockets) -> td.Box:
		"""Compute a lazy function for the point dipole source."""
		center = input_sockets['Center']
		temporal_shape = input_sockets['Temporal Shape']
		interpolate = input_sockets['Interpolate']

		has_center = not ct.FlowSignal.check(center)
		has_temporal_shape = not ct.FlowSignal.check(temporal_shape)
		has_interpolate = not ct.FlowSignal.check(interpolate)

		if has_temporal_shape and has_center and has_interpolate:
			pol = props['pol']
			return (
				center.scale_to_unit_system(ct.UNITS_TIDY3D)
				| temporal_shape
				| interpolate
			).compose_within(
				enclosing_func=lambda els: td.PointDipole(
					center=els[0].flatten().tolist(),
					source_time=els[1],
					interpolate=els[2],
					polarization=pol.name,
				)
			)
		return ct.FlowSignal.FlowPending

	####################
	# - FlowKind.Params
	####################
	@events.computes_output_socket(
		'Source',
		kind=ct.FlowKind.Params,
		# Loaded
		input_sockets={'Temporal Shape', 'Center', 'Interpolate'},
		input_socket_kinds={
			'Temporal Shape': ct.FlowKind.Params,
			'Center': ct.FlowKind.Params,
			'Interpolate': ct.FlowKind.Params,
		},
	)
	def compute_params(
		self,
		input_sockets,
	) -> td.PointDipole | ct.FlowSignal:
		"""Compute the point dipole source, given that all inputs are non-symbolic."""
		center = input_sockets['Center']
		temporal_shape = input_sockets['Temporal Shape']
		interpolate = input_sockets['Interpolate']

		has_center = not ct.FlowSignal.check(center)
		has_temporal_shape = not ct.FlowSignal.check(temporal_shape)
		has_interpolate = not ct.FlowSignal.check(interpolate)

		if has_center and has_temporal_shape and has_interpolate:
			return center | temporal_shape | interpolate
		return ct.FlowSignal.FlowPending

	####################
	# - Preview
	####################
	@events.computes_output_socket(
		'Source',
		kind=ct.FlowKind.Previews,
		# Loaded
		props={'sim_node_name'},
		output_sockets={'Source'},
		output_socket_kinds={'Source': ct.FlowKind.Params},
	)
	def compute_previews(self, props, output_sockets):
		output_params = output_sockets['Source']
		has_output_params = not ct.FlowSignal.check(output_params)

		if has_output_params and not output_params.symbols:
			return ct.PreviewsFlow(bl_object_names={props['sim_node_name']})
		return ct.PreviewsFlow()

	@events.on_value_changed(
		# Trigger
		socket_name={'Center'},
		prop_name='pol',
		run_on_init=True,
		# Loaded
		managed_objs={'modifier'},
		props={'pol'},
		input_sockets={'Center'},
		output_sockets={'Source'},
		output_socket_kinds={'Source': ct.FlowKind.Params},
	)
	def on_previewable_changed(
		self, managed_objs, props, input_sockets, output_sockets
	) -> None:
		SFP = ct.SimFieldPols

		center = input_sockets['Center']
		output_params = output_sockets['Source']

		has_center = not ct.FlowSignal.check(center)
		has_output_params = not ct.FlowSignal.check(output_params)

		if has_center and has_output_params and not output_params.symbols:
			axis = {
				SFP.Ex: 0,
				SFP.Ey: 1,
				SFP.Ez: 2,
				SFP.Hx: 0,
				SFP.Hy: 1,
				SFP.Hz: 2,
			}[props['pol']]

			# Push Loose Input Values to GeoNodes Modifier
			managed_objs['modifier'].bl_modifier(
				'NODES',
				{
					'node_group': import_geonodes(GeoNodes.SourcePointDipole),
					'unit_system': ct.UNITS_BLENDER,
					'inputs': {
						'Axis': axis,
					},
				},
				location=spux.scale_to_unit_system(center, ct.UNITS_BLENDER),
			)


####################
# - Blender Registration
####################
BL_REGISTER = [
	PointDipoleSourceNode,
]
BL_NODES = {ct.NodeType.PointDipoleSource: (ct.NodeCategory.MAXWELLSIM_SOURCES)}
