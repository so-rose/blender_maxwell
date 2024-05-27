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
	# - Outputs
	####################
	@events.computes_output_socket(
		'Source',
		input_sockets={'Temporal Shape', 'Center', 'Interpolate'},
		props={'pol_axis'},
		unit_systems={'Tidy3DUnits': ct.UNITS_TIDY3D},
		scale_input_sockets={
			'Center': 'Tidy3DUnits',
		},
	)
	def compute_source(
		self,
		input_sockets: dict[str, typ.Any],
		props: dict[str, typ.Any],
		unit_systems: dict,
	) -> td.PointDipole:
		pol_axis = {
			ct.SimSpaceAxis.X: 'Ex',
			ct.SimSpaceAxis.Y: 'Ey',
			ct.SimSpaceAxis.Z: 'Ez',
		}[props['pol_axis']]

		return td.PointDipole(
			center=input_sockets['Center'],
			source_time=input_sockets['Temporal Shape'],
			interpolate=input_sockets['Interpolate'],
			polarization=pol_axis,
		)

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
