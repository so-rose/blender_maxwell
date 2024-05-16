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


class GaussianBeamSourceNode(base.MaxwellSimNode):
	"""A finite-extent angled source simulating a wave produced by lensed fibers, with focal properties similar to that of ideal lasers, and linear polarization.

	The amplitude envelope is a gaussian function, and the complex electric field is a well-defined frequency-dependent phasor with very few shape parameters.

	The only critical shape parameter is the **waist**: At a chosen "focus" distance, the width of the beam has a chosen radius.
	These properties are called "waist distance" and "waist radius".
	At all other points, the width of the beam has a well-defined hyperbolic relationship to the waist, when given the IOR-dependent Rayleigh length.

	- Tidy3D Documentation: <https://docs.flexcompute.com/projects/tidy3d/en/latest/api/_autosummary/tidy3d.GaussianBeam.html#tidy3d.GaussianBeam>
	- Mathematical Formalism: <https://en.wikipedia.org/wiki/Gaussian_beam>
	"""

	node_type = ct.NodeType.GaussianBeamSource
	bl_label = 'Gaussian Beam Source'
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
		'Size': sockets.ExprSocketDef(
			size=spux.NumberSize1D.Vec2,
			mathtype=spux.MathType.Real,
			physical_type=spux.PhysicalType.Length,
			default_value=sp.Matrix([1, 1]),
		),
		'Waist Dist': sockets.ExprSocketDef(
			mathtype=spux.MathType.Real,
			physical_type=spux.PhysicalType.Length,
			default_value=0.0,
		),
		'Waist Radius': sockets.ExprSocketDef(
			mathtype=spux.MathType.Real,
			physical_type=spux.PhysicalType.Length,
			default_value=1.0,
			abs_min=0.01,
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
	num_freqs: int = bl_cache.BLField(1, abs_min=1, soft_max=20, prop_ui=True)

	####################
	# - UI
	####################
	def draw_props(self, _: bpy.types.Context, layout: bpy.types.UILayout):
		layout.prop(self, self.blfields['injection_axis'], expand=True)
		layout.prop(self, self.blfields['injection_direction'], expand=True)
		# layout.prop(self, self.blfields['num_freqs'], text='f Points')
		## TODO: UI is a bit crowded already!

	####################
	# - Outputs
	####################
	@events.computes_output_socket(
		'Angled Source',
		props={'sim_node_name', 'injection_axis', 'injection_direction', 'num_freqs'},
		input_sockets={
			'Temporal Shape',
			'Center',
			'Size',
			'Waist Dist',
			'Waist Radius',
			'Spherical',
			'Pol ∡',
		},
		unit_systems={'Tidy3DUnits': ct.UNITS_TIDY3D},
		scale_input_sockets={
			'Center': 'Tidy3DUnits',
			'Size': 'Tidy3DUnits',
			'Waist Dist': 'Tidy3DUnits',
			'Waist Radius': 'Tidy3DUnits',
			'Spherical': 'Tidy3DUnits',
			'Pol ∡': 'Tidy3DUnits',
		},
	)
	def compute_source(self, props, input_sockets, unit_systems):
		size_2d = input_sockets['Size']
		size = {
			ct.SimSpaceAxis.X: (0, *size_2d),
			ct.SimSpaceAxis.Y: (size_2d[0], 0, size_2d[1]),
			ct.SimSpaceAxis.Z: (*size_2d, 0),
		}[props['injection_axis']]

		# Display the results
		return td.GaussianBeam(
			name=props['sim_node_name'],
			center=input_sockets['Center'],
			size=size,
			source_time=input_sockets['Temporal Shape'],
			num_freqs=props['num_freqs'],
			direction=props['injection_direction'].plus_or_minus,
			angle_theta=input_sockets['Spherical'][0],
			angle_phi=input_sockets['Spherical'][1],
			pol_angle=input_sockets['Pol ∡'],
			waist_radius=input_sockets['Waist Radius'],
			waist_distance=input_sockets['Waist Dist'],
			## NOTE: Waist is place at this signed dist along neg. direction
		)

	####################
	# - Preview - Changes to Input Sockets
	####################
	@events.on_value_changed(
		# Trigger
		prop_name='preview_active',
		# Loaded
		managed_objs={'modifier'},
		props={'preview_active'},
	)
	def on_preview_changed(self, managed_objs, props):
		if props['preview_active']:
			managed_objs['modifier'].show_preview()
		else:
			managed_objs['modifier'].hide_preview()

	@events.on_value_changed(
		# Trigger
		socket_name={
			'Center',
			'Size',
			'Waist Dist',
			'Waist Radius',
			'Spherical',
			'Pol ∡',
		},
		prop_name={'injection_axis', 'injection_direction'},
		run_on_init=True,
		# Loaded
		managed_objs={'modifier'},
		props={'injection_axis', 'injection_direction'},
		input_sockets={
			'Temporal Shape',
			'Center',
			'Size',
			'Waist Dist',
			'Waist Radius',
			'Spherical',
			'Pol ∡',
		},
		unit_systems={'BlenderUnits': ct.UNITS_BLENDER},
		scale_input_sockets={
			'Center': 'BlenderUnits',
		},
	)
	def on_inputs_changed(self, managed_objs, props, input_sockets, unit_systems):
		size_2d = input_sockets['Size']
		size = {
			ct.SimSpaceAxis.X: sp.Matrix([0, *size_2d]),
			ct.SimSpaceAxis.Y: sp.Matrix([size_2d[0], 0, size_2d[1]]),
			ct.SimSpaceAxis.Z: sp.Matrix([*size_2d, 0]),
		}[props['injection_axis']]

		# Push Input Values to GeoNodes Modifier
		managed_objs['modifier'].bl_modifier(
			'NODES',
			{
				'node_group': import_geonodes(GeoNodes.SourceGaussianBeam),
				'unit_system': unit_systems['BlenderUnits'],
				'inputs': {
					# Orientation
					'Inj Axis': props['injection_axis'].axis,
					'Direction': props['injection_direction'].true_or_false,
					'theta': input_sockets['Spherical'][0],
					'phi': input_sockets['Spherical'][1],
					'Pol Angle': input_sockets['Pol ∡'],
					# Gaussian Beam
					'Size': size,
					'Waist Dist': input_sockets['Waist Dist'],
					'Waist Radius': input_sockets['Waist Radius'],
				},
			},
			location=input_sockets['Center'],
		)


####################
# - Blender Registration
####################
BL_REGISTER = [
	GaussianBeamSourceNode,
]
BL_NODES = {ct.NodeType.GaussianBeamSource: (ct.NodeCategory.MAXWELLSIM_SOURCES)}
