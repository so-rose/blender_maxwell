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
		'Angled Source': sockets.MaxwellSourceSocketDef(active_kind=ct.FlowKind.Func),
	}

	managed_obj_types: typ.ClassVar = {
		'modifier': managed_objs.ManagedBLModifier,
	}

	####################
	# - Properties
	####################
	injection_axis: ct.SimSpaceAxis = bl_cache.BLField(ct.SimSpaceAxis.X)
	injection_direction: ct.SimAxisDir = bl_cache.BLField(ct.SimAxisDir.Plus)

	####################
	# - UI
	####################
	def draw_props(self, _: bpy.types.Context, layout: bpy.types.UILayout):
		layout.prop(self, self.blfields['injection_axis'], expand=True)
		layout.prop(self, self.blfields['injection_direction'], expand=True)

	####################
	# - FlowKind.Value
	####################
	@events.computes_output_socket(
		'Angled Source',
		kind=ct.FlowKind.Value,
		# Loaded
		output_sockets={'Angled Source'},
		output_socket_kinds={'Angled Source': {ct.FlowKind.Func, ct.FlowKind.Params}},
	)
	def compute_value(self, output_sockets) -> ct.ParamsFlow | ct.FlowSignal:
		"""Compute the particular value of the simulation domain from strictly non-symbolic inputs."""
		output_func = output_sockets['Angled Source'][ct.FlowKind.Func]
		output_params = output_sockets['Angled Source'][ct.FlowKind.Params]

		has_output_func = not ct.FlowSignal.check(output_func)
		has_output_params = not ct.FlowSignal.check(output_params)

		if has_output_func and has_output_params and not output_params.symbols:
			return output_func.realize(output_params, disallow_jax=True)
		return ct.FlowSignal.FlowPending

	####################
	# - FlowKind.Func
	####################
	@events.computes_output_socket(
		'Angled Source',
		kind=ct.FlowKind.Func,
		# Loaded
		props={'sim_node_name', 'injection_axis', 'injection_direction'},
		input_sockets={'Temporal Shape', 'Center', 'Spherical', 'Pol ∡'},
		input_socket_kinds={
			'Temporal Shape': ct.FlowKind.Func,
			'Center': ct.FlowKind.Func,
			'Spherical': ct.FlowKind.Func,
			'Pol ∡': ct.FlowKind.Func,
		},
	)
	def compute_func(self, props, input_sockets) -> None:
		center = input_sockets['Center']
		temporal_shape = input_sockets['Temporal Shape']
		spherical = input_sockets['Spherical']
		pol_ang = input_sockets['Pol ∡']

		has_center = not ct.FlowSignal.check(center)
		has_temporal_shape = not ct.FlowSignal.check(temporal_shape)
		has_spherical = not ct.FlowSignal.check(spherical)
		has_pol_ang = not ct.FlowSignal.check(pol_ang)

		if has_center and has_temporal_shape and has_spherical and has_pol_ang:
			name = props['sim_node_name']
			inj_dir = props['injection_direction'].plus_or_minus
			size = {
				ct.SimSpaceAxis.X: (0, td.inf, td.inf),
				ct.SimSpaceAxis.Y: (td.inf, 0, td.inf),
				ct.SimSpaceAxis.Z: (td.inf, td.inf, 0),
			}[props['injection_axis']]

			return (
				center.scale_to_unit_system(ct.UNITS_TIDY3D)
				| temporal_shape
				| spherical.scale_to_unit_system(ct.UNITS_TIDY3D)
				| pol_ang.scale_to_unit_system(ct.UNITS_TIDY3D)
			).compose_within(
				lambda els: td.PlaneWave(
					name=name,
					center=els[0].flatten().tolist(),
					size=size,
					source_time=els[1],
					direction=inj_dir,
					angle_theta=els[2][0].item(0),
					angle_phi=els[2][1].item(0),
					pol_angle=els[3],
				)
			)
		return ct.FlowSignal.FlowPending

	####################
	# - FlowKind.Params
	####################
	@events.computes_output_socket(
		'Angled Source',
		kind=ct.FlowKind.Params,
		# Loaded
		input_sockets={'Temporal Shape', 'Center', 'Spherical', 'Pol ∡'},
		input_socket_kinds={
			'Temporal Shape': ct.FlowKind.Params,
			'Center': ct.FlowKind.Params,
			'Spherical': ct.FlowKind.Params,
			'Pol ∡': ct.FlowKind.Params,
		},
	)
	def compute_params(self, input_sockets) -> None:
		center = input_sockets['Center']
		temporal_shape = input_sockets['Temporal Shape']
		spherical = input_sockets['Spherical']
		pol_ang = input_sockets['Pol ∡']

		has_center = not ct.FlowSignal.check(center)
		has_temporal_shape = not ct.FlowSignal.check(temporal_shape)
		has_spherical = not ct.FlowSignal.check(spherical)
		has_pol_ang = not ct.FlowSignal.check(pol_ang)

		if has_center and has_temporal_shape and has_spherical and has_pol_ang:
			return center | temporal_shape | spherical | pol_ang
		return ct.FlowSignal.FlowPending

	####################
	# - Preview - Changes to Input Sockets
	####################
	@events.computes_output_socket(
		'Angled Source',
		kind=ct.FlowKind.Previews,
		# Loaded
		props={'sim_node_name'},
		output_sockets={'Angled Source'},
		output_socket_kinds={'Angled Source': ct.FlowKind.Params},
	)
	def compute_previews(self, props, output_sockets):
		output_params = output_sockets['Angled Source']
		has_output_params = not ct.FlowSignal.check(output_params)

		if has_output_params and not output_params.symbols:
			return ct.PreviewsFlow(bl_object_names={props['sim_node_name']})
		return ct.PreviewsFlow()

	@events.on_value_changed(
		# Trigger
		socket_name={'Center', 'Spherical', 'Pol ∡'},
		prop_name={'injection_axis', 'injection_direction'},
		run_on_init=True,
		# Loaded
		managed_objs={'modifier'},
		props={'injection_axis', 'injection_direction'},
		input_sockets={'Temporal Shape', 'Center', 'Spherical', 'Pol ∡'},
		output_sockets={'Angled Source'},
		output_socket_kinds={'Angled Source': ct.FlowKind.Params},
	)
	def on_previewable_changed(
		self, managed_objs, props, input_sockets, output_sockets
	):
		center = input_sockets['Center']
		spherical = input_sockets['Spherical']
		pol_ang = input_sockets['Pol ∡']
		output_params = output_sockets['Angled Source']

		has_center = not ct.FlowSignal.check(center)
		has_spherical = not ct.FlowSignal.check(spherical)
		has_pol_ang = not ct.FlowSignal.check(pol_ang)
		has_output_params = not ct.FlowSignal.check(output_params)

		if (
			has_center
			and has_spherical
			and has_pol_ang
			and has_output_params
			and not output_params.symbols
		):
			# Push Input Values to GeoNodes Modifier
			managed_objs['modifier'].bl_modifier(
				'NODES',
				{
					'node_group': import_geonodes(GeoNodes.SourcePlaneWave),
					'unit_system': ct.UNITS_BLENDER,
					'inputs': {
						'Inj Axis': props['injection_axis'].axis,
						'Direction': props['injection_direction'].true_or_false,
						'theta': spherical[0],
						'phi': spherical[1],
						'Pol Angle': pol_ang,
					},
				},
				location=spux.scale_to_unit_system(center, ct.UNITS_BLENDER),
			)


####################
# - Blender Registration
####################
BL_REGISTER = [
	PlaneWaveSourceNode,
]
BL_NODES = {ct.NodeType.PlaneWaveSource: (ct.NodeCategory.MAXWELLSIM_SOURCES)}
