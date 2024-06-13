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

"""Implements `PointDipoleSourceNode`."""

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


class PointDipoleSourceNode(base.MaxwellSimNode):
	"""A point dipole with E or H oriented linear polarization along an axis-aligned angle."""

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
			mathtype=MT.Real,
			physical_type=PT.Length,
			default_value=sp.ImmutableMatrix([0, 0, 0]),
		),
		'Interpolate': sockets.BoolSocketDef(
			default_value=True,
		),
	}
	output_sockets: typ.ClassVar = {
		'Source': sockets.MaxwellSourceSocketDef(active_kind=FK.Func),
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
		"""Draw choices of polarization direction."""
		layout.prop(self, self.blfields['pol'], expand=True)

	####################
	# - FlowKind.Value
	####################
	@events.computes_output_socket(
		'Source',
		kind=FK.Value,
		# Loaded
		output_sockets={'Source'},
		output_socket_kinds={'Source': {FK.Func, FK.Params}},
	)
	def compute_value(self, output_sockets) -> ct.ParamsFlow | FS:
		"""Compute the particular value of the simulation domain from strictly non-symbolic inputs."""
		value = events.realize_known(output_sockets['Source'])
		if value is not None:
			return value
		return FS.FlowPending

	####################
	# - FlowKind.Func
	####################
	@events.computes_output_socket(
		'Source',
		kind=FK.Func,
		# Loaded
		props={'pol'},
		inscks_kinds={
			'Temporal Shape': FK.Func,
			'Center': FK.Func,
			'Interpolate': FK.Func,
		},
		scale_input_sockets={
			'Center': ct.UNITS_TIDY3D,
		},
	)
	def compute_func(self, props, input_sockets) -> td.Box:
		"""Compute a lazy function for the point dipole source."""
		center = input_sockets['Center']
		temporal_shape = input_sockets['Temporal Shape']
		interpolate = input_sockets['Interpolate']

		pol = props['pol']

		return (center | temporal_shape | interpolate).compose_within(
			lambda els: td.PointDipole(
				center=els[0].flatten().tolist(),
				source_time=els[1],
				interpolate=els[2],
				polarization=pol.name,
			)
		)
		return FS.FlowPending

	####################
	# - FlowKind.Params
	####################
	@events.computes_output_socket(
		'Source',
		kind=FK.Params,
		# Loaded
		input_sockets={'Temporal Shape', 'Center', 'Interpolate'},
		input_socket_kinds={
			'Temporal Shape': FK.Params,
			'Center': FK.Params,
			'Interpolate': FK.Params,
		},
	)
	def compute_params(
		self,
		input_sockets,
	) -> td.PointDipole:
		"""Compute the function parameters of the lazy function."""
		center = input_sockets['Center']
		temporal_shape = input_sockets['Temporal Shape']
		interpolate = input_sockets['Interpolate']

		return center | temporal_shape | interpolate

	####################
	# - Preview
	####################
	@events.computes_output_socket(
		'Source',
		kind=FK.Previews,
		# Loaded
		props={'sim_node_name'},
	)
	def compute_previews(self, props):
		"""Mark the point dipole as participating in the 3D preview."""
		return ct.PreviewsFlow(bl_object_names={props['sim_node_name']})

	@events.on_value_changed(
		# Trigger
		socket_name={'Center': {FK.Func, FK.Params}},
		prop_name='pol',
		run_on_init=True,
		# Loaded
		managed_objs={'modifier'},
		props={'pol'},
		inscks_kinds={
			'Center': {FK.Func, FK.Params},
		},
		scale_input_sockets={
			'Center': ct.UNITS_BLENDER,
		},
	)
	def on_previewable_changed(self, managed_objs, props, input_sockets) -> None:
		"""Push changes in the inputs to the pol/center."""
		SFP = ct.SimFieldPols

		center = events.realize_preview(input_sockets['Center'])
		axis = {
			SFP.Ex: 0,
			SFP.Ey: 1,
			SFP.Ez: 2,
			SFP.Hx: 0,
			SFP.Hy: 1,
			SFP.Hz: 2,
		}[props['pol']]

		managed_objs['modifier'].bl_modifier(
			'NODES',
			{
				'node_group': import_geonodes(GeoNodes.SourcePointDipole),
				'inputs': {
					'Axis': axis,
				},
			},
			location=center,
		)


####################
# - Blender Registration
####################
BL_REGISTER = [
	PointDipoleSourceNode,
]
BL_NODES = {ct.NodeType.PointDipoleSource: (ct.NodeCategory.MAXWELLSIM_SOURCES)}
