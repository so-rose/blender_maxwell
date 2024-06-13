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

"""Implements `GaussianBeamSourceNode`."""

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
FK = ct.FlowKind
FS = ct.FlowSignal


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
			default_value=sp.ImmutableMatrix([0, 0, 0]),
		),
		'Size': sockets.ExprSocketDef(
			size=spux.NumberSize1D.Vec2,
			mathtype=spux.MathType.Real,
			physical_type=spux.PhysicalType.Length,
			default_value=sp.ImmutableMatrix([1, 1]),
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
			default_value=sp.ImmutableMatrix([0, 0]),
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
	injection_axis: ct.SimSpaceAxis = bl_cache.BLField(ct.SimSpaceAxis.X)
	injection_direction: ct.SimAxisDir = bl_cache.BLField(ct.SimAxisDir.Plus)
	num_freqs: int = bl_cache.BLField(1, abs_min=1, soft_max=20)

	####################
	# - UI
	####################
	def draw_props(self, _: bpy.types.Context, layout: bpy.types.UILayout):
		"""Draw choices of injection axis and direction."""
		layout.prop(self, self.blfields['injection_axis'], expand=True)
		layout.prop(self, self.blfields['injection_direction'], expand=True)
		# layout.prop(self, self.blfields['num_freqs'], text='f Points')
		## TODO: UI is a bit crowded already!

	####################
	# - FlowKind.Value
	####################
	@events.computes_output_socket(
		'Angled Source',
		kind=FK.Value,
		# Loaded
		outscks_kinds={
			'Angled Source': {FK.Func, FK.Params},
		},
	)
	def compute_value(self, output_sockets) -> ct.ParamsFlow | FS:
		"""Compute the particular value of the simulation domain from strictly non-symbolic inputs."""
		output_func = output_sockets['Structure'][FK.Func]
		output_params = output_sockets['Structure'][FK.Params]

		if not output_params.symbols:
			return output_func.realize(output_params, disallow_jax=True)
		return ct.FlowSignal.FlowPending

	####################
	# - FlowKind.Value
	####################
	@events.computes_output_socket(
		'Angled Source',
		kind=FK.Func,
		# Loaded
		props={'sim_node_name', 'injection_axis', 'injection_direction', 'num_freqs'},
		inscks_kinds={
			'Temporal Shape': FK.Func,
			'Center': FK.Func,
			'Size': FK.Func,
			'Waist Dist': FK.Func,
			'Waist Radius': FK.Func,
			'Spherical': FK.Func,
			'Pol ∡': FK.Func,
		},
		scale_input_sockets={
			'Center': ct.UNITS_TIDY3D,
			'Size': ct.UNITS_TIDY3D,
			'Waist Dist': ct.UNITS_TIDY3D,
			'Waist Radius': ct.UNITS_TIDY3D,
			'Spherical': ct.UNITS_TIDY3D,
			'Pol ∡': ct.UNITS_TIDY3D,
		},
	)
	def compute_func(self, props, input_sockets) -> td.GaussianBeam:
		"""Compute a function that returns a gaussian beam object, given sufficient parameters."""
		inj_dir = props['injection_axis']

		def size(size_2d: tuple[float, float]) -> tuple[float, float, float]:
			return {
				ct.SimSpaceAxis.X: (0, *size_2d),
				ct.SimSpaceAxis.Y: (size_2d[0], 0, size_2d[1]),
				ct.SimSpaceAxis.Z: (*size_2d, 0),
			}[inj_dir]

		center = input_sockets['Center']
		size_2d = input_sockets['Size']
		temporal_shape = input_sockets['Temporal Shape']
		spherical = input_sockets['Spherical']
		pol_ang = input_sockets['Pol ∡']
		waist_radius = input_sockets['Waist Radius']
		waist_dist = input_sockets['Waist Dist']

		sim_node_name = props['sim_node_name']
		num_freqs = props['num_freqs']
		return (
			center
			| size_2d
			| temporal_shape
			| spherical
			| pol_ang
			| waist_radius
			| waist_dist
		).compose_within(
			lambda els: td.GaussianBeam(
				name=sim_node_name,
				center=els[0],
				size=size(els[1]),
				source_time=els[2],
				num_freqs=num_freqs,
				direction=inj_dir.plus_or_minus,
				angle_theta=els[3].item(0),
				angle_phi=els[3].item(1),
				pol_angle=els[4],
				waist_radius=els[5],
				waist_distance=els[6],
			)
		)

	####################
	# - FlowKind.Params
	####################
	@events.computes_output_socket(
		'Structure',
		kind=FK.Params,
		# Loaded
		inscks_kinds={
			'Temporal Shape': FK.Func,
			'Center': FK.Func,
			'Size': FK.Func,
			'Waist Dist': FK.Func,
			'Waist Radius': FK.Func,
			'Spherical': FK.Func,
			'Pol ∡': FK.Func,
		},
	)
	def compute_params(self, input_sockets) -> ct.ParamsFlow:
		"""Propagate function parameters from inputs."""
		center = input_sockets['Center']
		size_2d = input_sockets['Size']
		temporal_shape = input_sockets['Temporal Shape']
		spherical = input_sockets['Spherical']
		pol_ang = input_sockets['Pol ∡']
		waist_radius = input_sockets['Waist Radius']
		waist_dist = input_sockets['Waist Dist']

		return (
			center
			| size_2d
			| temporal_shape
			| spherical
			| pol_ang
			| waist_radius
			| waist_dist
		)

	####################
	# - Events: Preview
	####################
	@events.computes_output_socket(
		'Angled Source',
		kind=FK.Previews,
		# Loaded
		props={'sim_node_name'},
		outscks_kinds={'Angled Source': ct.FlowKind.Func},
		output_sockets_optional={'Angled Source'},
	)
	def compute_previews(self, props, output_sockets):
		"""Update the preview state when the name or output function change."""
		if not FS.check(output_sockets['Structure']):
			return ct.PreviewsFlow(bl_object_names={props['sim_node_name']})
		return ct.PreviewsFlow()

	@events.on_value_changed(
		# Trigger
		socket_name={
			'Center': {FK.Func, FK.Params},
			'Size': {FK.Func, FK.Params},
			'Waist Dist': {FK.Func, FK.Params},
			'Waist Radius': {FK.Func, FK.Params},
			'Spherical': {FK.Func, FK.Params},
			'Pol ∡': {FK.Func, FK.Params},
		},
		prop_name={'injection_axis', 'injection_direction'},
		run_on_init=True,
		# Loaded
		managed_objs={'modifier'},
		props={'injection_axis', 'injection_direction'},
		inscks_kinds={
			'Center': {FK.Func, FK.Params},
			'Size': {FK.Func, FK.Params},
			'Waist Dist': {FK.Func, FK.Params},
			'Waist Radius': {FK.Func, FK.Params},
			'Spherical': {FK.Func, FK.Params},
			'Pol ∡': {FK.Func, FK.Params},
		},
		scale_input_sockets={
			'Center': ct.UNITS_BLENDER,
			'Size': ct.UNITS_BLENDER,
			'Waist Dist': ct.UNITS_BLENDER,
			'Waist Radius': ct.UNITS_BLENDER,
			'Spherical': ct.UNITS_BLENDER,
			'Pol ∡': ct.UNITS_BLENDER,
		},
	)
	def on_previewable_chnaged(self, managed_objs, props, input_sockets):
		"""Update the preview when relevant inputs change."""
		center = events.realize_preview(input_sockets['Center'])
		size_2d = events.realize_preview(input_sockets['Size'])
		spherical = events.realize_preview(input_sockets['Spherical'])
		pol_ang = events.realize_preview(input_sockets['Pol ∡'])
		waist_radius = events.realize_preview(input_sockets['Waist Radius'])
		waist_dist = events.realize_preview(input_sockets['Waist Dist'])

		# Retrieve Properties
		inj_dir = props['injection_axis']
		size = {
			ct.SimSpaceAxis.X: sp.ImmutableMatrix([0, *size_2d]),
			ct.SimSpaceAxis.Y: sp.ImmutableMatrix([size_2d[0], 0, size_2d[1]]),
			ct.SimSpaceAxis.Z: sp.ImmutableMatrix([*size_2d, 0]),
		}[props['injection_axis']]

		# Push Updated Values
		managed_objs['modifier'].bl_modifier(
			'NODES',
			{
				'node_group': import_geonodes(GeoNodes.SourceGaussianBeam),
				'inputs': {
					# Orientation
					'Inj Axis': inj_dir.axis,
					'Direction': inj_dir.true_or_false,
					'theta': spherical[0],
					'phi': spherical[1],
					'Pol Angle': pol_ang,
					# Gaussian Beam
					'Size': size,
					'Waist Radius': waist_radius,
					'Waist Dist': waist_dist,
				},
			},
			location=center,
		)


####################
# - Blender Registration
####################
BL_REGISTER = [
	GaussianBeamSourceNode,
]
BL_NODES = {ct.NodeType.GaussianBeamSource: (ct.NodeCategory.MAXWELLSIM_SOURCES)}
