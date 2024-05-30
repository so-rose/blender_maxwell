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
from blender_maxwell.utils import extra_sympy_units as spux

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
		'Source': sockets.MaxwellSourceSocketDef(),
	}

	managed_obj_types: typ.ClassVar = {
		'modifier': managed_objs.ManagedBLModifier,
	}

	####################
	# - Properties
	####################
	pol_axis: ct.SimSpaceAxis = bl_cache.BLField(ct.SimSpaceAxis.X, prop_ui=True)

	####################
	# - UI
	####################
	def draw_props(self, _: bpy.types.Context, layout: bpy.types.UILayout):
		layout.prop(self, self.blfields['pol_axis'], expand=True)

	####################
	# - FlowKind.Value
	####################
	@events.computes_output_socket(
		'Source',
		# Loaded
		props={'pol_axis'},
		input_sockets={'Temporal Shape', 'Center', 'Interpolate'},
		output_sockets={'Source'},
		output_socket_kinds={'Source': ct.FlowKind.Params},
	)
	def compute_source_value(
		self, input_sockets, props, output_sockets
	) -> td.PointDipole | ct.FlowSignal:
		"""Compute the point dipole source, given that all inputs are non-symbolic."""
		temporal_shape = input_sockets['Temporal Shape']
		center = input_sockets['Center']
		interpolate = input_sockets['Interpolate']
		output_params = output_sockets['Source']

		has_temporal_shape = not ct.FlowSignal.check(temporal_shape)
		has_center = not ct.FlowSignal.check(center)
		has_interpolate = not ct.FlowSignal.check(interpolate)
		has_output_params = not ct.FlowSignal.check(output_params)

		if (
			has_temporal_shape
			and has_center
			and has_interpolate
			and has_output_params
			and not output_params.symbols
		):
			pol_axis = {
				ct.SimSpaceAxis.X: 'Ex',
				ct.SimSpaceAxis.Y: 'Ey',
				ct.SimSpaceAxis.Z: 'Ez',
			}[props['pol_axis']]
			## TODO: Need Hx, Hy, Hz too?

			return td.PointDipole(
				center=spux.convert_to_unit_system(center, ct.UNITS_TIDY3D),
				source_time=temporal_shape,
				interpolate=interpolate,
				polarization=pol_axis,
			)
		return ct.FlowSignal.FlowPending

	####################
	# - FlowKind.Func
	####################
	@events.computes_output_socket(
		'Source',
		kind=ct.FlowKind.Func,
		# Loaded
		props={'pol_axis'},
		input_sockets={'Temporal Shape', 'Center', 'Interpolate'},
		input_socket_kinds={
			'Temporal Shape': ct.FlowKind.Func,
			'Center': ct.FlowKind.Func,
			'Interpolate': ct.FlowKind.Func,
		},
		output_sockets={'Source'},
		output_socket_kinds={'Source': ct.FlowKind.Params},
	)
	def compute_source_func(self, props, input_sockets, output_sockets) -> td.Box:
		"""Compute a lazy function for the point dipole source."""
		center = input_sockets['Center']
		temporal_shape = input_sockets['Temporal Shape']
		interpolate = input_sockets['Interpolate']
		output_params = output_sockets['Source']

		has_center = not ct.FlowSignal.check(center)
		has_temporal_shape = not ct.FlowSignal.check(temporal_shape)
		has_interpolate = not ct.FlowSignal.check(interpolate)
		has_output_params = not ct.FlowSignal.check(output_params)

		if has_temporal_shape and has_center and has_interpolate and has_output_params:
			pol_axis = {
				ct.SimSpaceAxis.X: 'Ex',
				ct.SimSpaceAxis.Y: 'Ey',
				ct.SimSpaceAxis.Z: 'Ez',
			}[props['pol_axis']]
			## TODO: Need Hx, Hy, Hz too?

			return (center | temporal_shape | interpolate).compose_within(
				enclosing_func=lambda els: td.PointDipole(
					center=els[0],
					source_time=els[1],
					interpolate=els[2],
					polarization=pol_axis,
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
		temporal_shape = input_sockets['Temporal Shape']
		center = input_sockets['Center']
		interpolate = input_sockets['Interpolate']

		has_temporal_shape = not ct.FlowSignal.check(temporal_shape)
		has_center = not ct.FlowSignal.check(center)
		has_interpolate = not ct.FlowSignal.check(interpolate)

		if has_temporal_shape and has_center and has_interpolate:
			return temporal_shape | center | interpolate
		return ct.FlowSignal.FlowPending

	####################
	# - Preview
	####################
	@events.computes_output_socket(
		'Source',
		kind=ct.FlowKind.Previews,
		# Loaded
		props={'sim_node_name'},
	)
	def compute_previews(self, props):
		return ct.PreviewsFlow(bl_object_names={props['sim_node_name']})

	@events.on_value_changed(
		socket_name={'Center'},
		prop_name='pol_axis',
		run_on_init=True,
		# Pass Data
		managed_objs={'modifier'},
		props={'pol_axis'},
		input_sockets={'Center'},
		unit_systems={'BlenderUnits': ct.UNITS_BLENDER},
		scale_input_sockets={'Center': 'BlenderUnits'},
	)
	def on_inputs_changed(
		self, managed_objs, props, input_sockets, unit_systems
	) -> None:
		modifier = managed_objs['modifier']
		unit_system = unit_systems['BlenderUnits']
		axis = {
			ct.SimSpaceAxis.X: 0,
			ct.SimSpaceAxis.Y: 1,
			ct.SimSpaceAxis.Z: 2,
		}[props['pol_axis']]

		# Push Loose Input Values to GeoNodes Modifier
		modifier.bl_modifier(
			'NODES',
			{
				'node_group': import_geonodes(GeoNodes.SourcePointDipole),
				'inputs': {'Axis': axis},
				'unit_system': unit_system,
			},
			location=input_sockets['Center'],
		)


####################
# - Blender Registration
####################
BL_REGISTER = [
	PointDipoleSourceNode,
]
BL_NODES = {ct.NodeType.PointDipoleSource: (ct.NodeCategory.MAXWELLSIM_SOURCES)}
