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


class PlaneWaveSourceNode(base.MaxwellSimNode):
	node_type = ct.NodeType.PlaneWaveSource
	bl_label = 'Plane Wave Source'
	use_sim_node_name = True

	####################
	# - Sockets
	####################
	input_sockets: typ.ClassVar = {
		'Temporal Shape': sockets.MaxwellTemporalShapeSocketDef(),
		'Center': sockets.ExprSocketDef(
			shape=(3,),
			mathtype=spux.MathType.Real,
			physical_type=spux.PhysicalType.Length,
			default_value=sp.Matrix([0, 0, 0]),
		),
		'Spherical': sockets.ExprSocketDef(
			shape=(2,),
			mathtype=spux.MathType.Real,
			physical_type=spux.PhysicalType.Angle,
			default_value=sp.Matrix([0, 0]),
		),
		'Pol ∡': sockets.ExprSocketDef(
			physical_type=spux.PhysicalType.Angle,
			default_value=0,
		),
	}
	output_sockets: typ.ClassVar = {
		'Angled Source': sockets.MaxwellSourceSocketDef(),
	}

	managed_obj_types: typ.ClassVar = {
		'mesh': managed_objs.ManagedBLMesh,
		'modifier': managed_objs.ManagedBLModifier,
	}

	####################
	# - Properties
	####################
	injection_axis: ct.SimSpaceAxis = bl_cache.BLField(ct.SimSpaceAxis.X, prop_ui=True)
	injection_direction: ct.SimAxisDir = bl_cache.BLField(
		ct.SimAxisDir.Plus, prop_ui=True
	)

	####################
	# - UI
	####################
	def draw_props(self, _: bpy.types.Context, layout: bpy.types.UILayout):
		layout.prop(self, self.blfields['injection_axis'], expand=True)
		layout.prop(self, self.blfields['injection_direction'], expand=True)

	####################
	# - Output Socket Computation
	####################
	@events.computes_output_socket(
		'Angled Source',
		props={'sim_node_name', 'injection_axis', 'injection_direction'},
		input_sockets={'Temporal Shape', 'Center', 'Spherical', 'Pol ∡'},
		unit_systems={'Tidy3DUnits': ct.UNITS_TIDY3D},
		scale_input_sockets={
			'Center': 'Tidy3DUnits',
			'Spherical': 'Tidy3DUnits',
			'Pol ∡': 'Tidy3DUnits',
		},
	)
	def compute_source(self, props, input_sockets, unit_systems):
		size = {
			ct.SimSpaceAxis.X: (0, td.inf, td.inf),
			ct.SimSpaceAxis.Y: (td.inf, 0, td.inf),
			ct.SimSpaceAxis.Z: (td.inf, td.inf, 0),
		}[props['injection_axis']]

		# Display the results
		return td.PlaneWave(
			name=props['sim_node_name'],
			center=input_sockets['Center'],
			size=size,
			source_time=input_sockets['Temporal Shape'],
			direction=props['injection_direction'].plus_or_minus,
			angle_theta=input_sockets['Spherical'][0],
			angle_phi=input_sockets['Spherical'][1],
			pol_angle=input_sockets['Pol ∡'],
		)

	####################
	# - Preview - Changes to Input Sockets
	####################
	@events.on_value_changed(
		# Trigger
		prop_name='preview_active',
		# Loaded
		managed_objs={'mesh'},
		props={'preview_active'},
	)
	def on_preview_changed(self, managed_objs, props):
		"""Enables/disables previewing of the GeoNodes-driven mesh, regardless of whether a particular GeoNodes tree is chosen."""
		mesh = managed_objs['mesh']

		# Push Preview State to Managed Mesh
		if props['preview_active']:
			mesh.show_preview()
		else:
			mesh.hide_preview()

	@events.on_value_changed(
		# Trigger
		socket_name={'Center', 'Spherical', 'Pol ∡'},
		prop_name={'injection_axis', 'injection_direction'},
		run_on_init=True,
		# Loaded
		managed_objs={'mesh', 'modifier'},
		props={'injection_axis', 'injection_direction'},
		input_sockets={'Temporal Shape', 'Center', 'Spherical', 'Pol ∡'},
		unit_systems={'BlenderUnits': ct.UNITS_BLENDER},
		scale_input_sockets={
			'Center': 'BlenderUnits',
		},
	)
	def on_inputs_changed(self, managed_objs, props, input_sockets, unit_systems):
		# Push Input Values to GeoNodes Modifier
		managed_objs['modifier'].bl_modifier(
			managed_objs['mesh'].bl_object(location=input_sockets['Center']),
			'NODES',
			{
				'node_group': import_geonodes(GeoNodes.SourcePlaneWave),
				'unit_system': unit_systems['BlenderUnits'],
				'inputs': {
					'Inj Axis': props['injection_axis'].axis,
					'Direction': props['injection_direction'].true_or_false,
					'theta': input_sockets['Spherical'][0],
					'phi': input_sockets['Spherical'][1],
					'Pol Angle': input_sockets['Pol ∡'],
				},
			},
		)


####################
# - Blender Registration
####################
BL_REGISTER = [
	PlaneWaveSourceNode,
]
BL_NODES = {ct.NodeType.PlaneWaveSource: (ct.NodeCategory.MAXWELLSIM_SOURCES)}
