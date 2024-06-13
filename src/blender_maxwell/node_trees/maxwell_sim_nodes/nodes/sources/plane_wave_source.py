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

"""Implements `PlaneWaveSourceNode`."""

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
MT = spux.MathType
PT = spux.PhysicalType


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
		'Temporal Shape': sockets.MaxwellTemporalShapeSocketDef(active_kind=FK.Func),
		'Center': sockets.ExprSocketDef(
			size=spux.NumberSize1D.Vec3,
			mathtype=MT.Real,
			physical_type=PT.Length,
			default_value=sp.ImmutableMatrix([0, 0, 0]),
		),
		'Spherical': sockets.ExprSocketDef(
			size=spux.NumberSize1D.Vec2,
			mathtype=MT.Real,
			physical_type=PT.Angle,
			default_value=sp.ImmutableMatrix([0, 0]),
		),
		'Pol ∡': sockets.ExprSocketDef(
			physical_type=PT.Angle,
			default_value=0,
		),
	}
	output_sockets: typ.ClassVar = {
		'Angled Source': sockets.MaxwellSourceSocketDef(active_kind=FK.Func),
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
		"""Draw choices of injection axis and direction."""
		layout.prop(self, self.blfields['injection_axis'], expand=True)
		layout.prop(self, self.blfields['injection_direction'], expand=True)

	####################
	# - FlowKind.Value
	####################
	@events.computes_output_socket(
		'Angled Source',
		kind=FK.Value,
		# Loaded
		outscks_kinds={'Angled Source': {FK.Func, FK.Params}},
	)
	def compute_value(self, output_sockets) -> ct.ParamsFlow | FS:
		"""Compute the particular value of the simulation domain from strictly non-symbolic inputs."""
		value = events.realize_known(output_sockets['Angled Source'])
		if value is not None:
			return value
		return FS.FlowPending

	####################
	# - FlowKind.Func
	####################
	@events.computes_output_socket(
		'Angled Source',
		kind=FK.Func,
		# Loaded
		props={'sim_node_name', 'injection_axis', 'injection_direction'},
		inscks_kinds={
			'Temporal Shape': FK.Func,
			'Center': FK.Func,
			'Spherical': FK.Func,
			'Pol ∡': FK.Func,
		},
		scale_input_sockets={
			'Center': ct.UNITS_TIDY3D,
			'Spherical': ct.UNITS_TIDY3D,
			'Pol ∡': ct.UNITS_TIDY3D,
		},
	)
	def compute_func(self, props, input_sockets) -> None:
		"""Compute a lazy function for the plane wave source."""
		center = input_sockets['Center']
		temporal_shape = input_sockets['Temporal Shape']
		spherical = input_sockets['Spherical']
		pol_ang = input_sockets['Pol ∡']

		name = props['sim_node_name']
		inj_dir = props['injection_direction'].plus_or_minus
		size = {
			ct.SimSpaceAxis.X: (0, td.inf, td.inf),
			ct.SimSpaceAxis.Y: (td.inf, 0, td.inf),
			ct.SimSpaceAxis.Z: (td.inf, td.inf, 0),
		}[props['injection_axis']]

		return (center | temporal_shape | spherical | pol_ang).compose_within(
			lambda els: td.PlaneWave(
				name=name,
				center=els[0].flatten().tolist(),
				size=size,
				source_time=els[1],
				direction=inj_dir,
				angle_theta=els[2].item(0),
				angle_phi=els[2].item(1),
				pol_angle=els[3],
			)
		)

	####################
	# - FlowKind.Params
	####################
	@events.computes_output_socket(
		'Angled Source',
		kind=FK.Params,
		# Loaded
		inscks_kinds={
			'Temporal Shape': FK.Params,
			'Center': FK.Params,
			'Spherical': FK.Params,
			'Pol ∡': FK.Params,
		},
	)
	def compute_params(self, input_sockets) -> None:
		"""Compute the function parameters of the lazy function."""
		center = input_sockets['Center']
		temporal_shape = input_sockets['Temporal Shape']
		spherical = input_sockets['Spherical']
		pol_ang = input_sockets['Pol ∡']

		return center | temporal_shape | spherical | pol_ang

	####################
	# - Preview - Changes to Input Sockets
	####################
	@events.computes_output_socket(
		'Angled Source',
		kind=FK.Previews,
		# Loaded
		props={'sim_node_name'},
	)
	def compute_previews(self, props):
		"""Mark the plane wave as participating in the 3D preview."""
		return ct.PreviewsFlow(bl_object_names={props['sim_node_name']})

	@events.on_value_changed(
		# Trigger
		socket_name={
			'Center': {FK.Func, FK.Params},
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
			'Spherical': {FK.Func, FK.Params},
			'Pol ∡': {FK.Func, FK.Params},
		},
		scale_input_sockets={
			'Center': ct.UNITS_BLENDER,
			'Spherical': ct.UNITS_BLENDER,
			'Pol ∡': ct.UNITS_BLENDER,
		},
	)
	def on_previewable_changed(self, managed_objs, props, input_sockets):
		"""Push changes in the inputs to the pol/center."""
		center = events.realize_preview(input_sockets['Center'])
		spherical = events.realize_preview(input_sockets['Spherical'])
		pol_ang = events.realize_preview(input_sockets['Pol ∡'])

		# Push Input Values to GeoNodes Modifier
		managed_objs['modifier'].bl_modifier(
			'NODES',
			{
				'node_group': import_geonodes(GeoNodes.SourcePlaneWave),
				'inputs': {
					'Inj Axis': props['injection_axis'].axis,
					'Direction': props['injection_direction'].true_or_false,
					'theta': spherical.item(0),
					'phi': spherical.item(1),
					'Pol Angle': pol_ang,
				},
			},
			location=center,
		)


####################
# - Blender Registration
####################
BL_REGISTER = [
	PlaneWaveSourceNode,
]
BL_NODES = {ct.NodeType.PlaneWaveSource: (ct.NodeCategory.MAXWELLSIM_SOURCES)}
