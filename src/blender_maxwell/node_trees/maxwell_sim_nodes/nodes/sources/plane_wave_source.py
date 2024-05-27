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


class PlaneWaveSourceNode(base.MaxwellSimNode):
	"""An infinite-extent angled source simulating an plane wave with linear polarization.

	The amplitude envelope is a gaussian function, and the complex electric field is a well-defined frequency-dependent phasor with very few shape parameters.

	The only critical shape parameter is the **waist**: At a chosen "focus" distance, the width of the beam has a chosen radius.
	These properties are called "waist distance" and "waist radius".
	At all other points, the width of the beam has a well-defined hyperbolic relationship to the waist, when given the IOR-dependent Rayleigh length.

	- Tidy3D Documentation: <https://docs.flexcompute.com/projects/tidy3d/en/latest/api/_autosummary/tidy3d.GaussianBeam.html#tidy3d.GaussianBeam>
	- Mathematical Formalism: <https://en.wikipedia.org/wiki/Gaussian_beam>
	"""

	node_type = ct.NodeType.PlaneWaveSource
	bl_label = 'Plane Wave Source'
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
		'Spherical': sockets.ExprSocketDef(
			size=spux.NumberSize1D.Vec2,
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
	@events.computes_output_socket(
		'Angled Source',
		kind=ct.FlowKind.Previews,
		# Loaded
		props={'sim_node_name'},
	)
	def compute_previews(self, props):
		return ct.PreviewsFlow(bl_object_names={props['sim_node_name']})

	@events.on_value_changed(
		# Trigger
		socket_name={'Center', 'Spherical', 'Pol ∡'},
		prop_name={'injection_axis', 'injection_direction'},
		run_on_init=True,
		# Loaded
		managed_objs={'modifier'},
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
			location=input_sockets['Center'],
		)


####################
# - Blender Registration
####################
BL_REGISTER = [
	PlaneWaveSourceNode,
]
BL_NODES = {ct.NodeType.PlaneWaveSource: (ct.NodeCategory.MAXWELLSIM_SOURCES)}
