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

"""Implements `SphereStructureNode`."""

import typing as typ

import sympy as sp
import sympy.physics.units as spu
import tidy3d as td

from blender_maxwell.assets.geonodes import GeoNodes, import_geonodes
from blender_maxwell.utils import logger
from blender_maxwell.utils import sympy_extra as spux

from .... import contracts as ct
from .... import managed_objs, sockets
from ... import base, events

log = logger.get(__name__)

FK = ct.FlowKind
FS = ct.FlowSignal
MT = spux.MathType


class SphereStructureNode(base.MaxwellSimNode):
	"""A generic sphere structure with configurable size and center."""

	node_type = ct.NodeType.SphereStructure
	bl_label = 'Sphere Structure'
	use_sim_node_name = True

	####################
	# - Sockets
	####################
	input_sockets: typ.ClassVar = {
		'Medium': sockets.MaxwellMediumSocketDef(),
		'Center': sockets.ExprSocketDef(
			size=spux.NumberSize1D.Vec3,
			default_unit=spu.micrometer,
			default_value=sp.ImmutableMatrix([0, 0, 0]),
		),
		'Radius': sockets.ExprSocketDef(
			default_unit=spu.nanometer,
			default_value=150,
		),
	}
	output_sockets: typ.ClassVar = {
		'Structure': sockets.MaxwellStructureSocketDef(active_kind=FK.Func),
	}

	managed_obj_types: typ.ClassVar = {
		'modifier': managed_objs.ManagedBLModifier,
	}

	####################
	# - FlowKind.Value
	####################
	@events.computes_output_socket(
		'Structure',
		kind=FK.Value,
		# Loaded
		outscks_kinds={
			'Structure': {FK.Func, FK.Params},
		},
	)
	def compute_value(self, output_sockets) -> ct.ParamsFlow | FS:
		"""Compute the particular value of the simulation domain from strictly non-symbolic inputs."""
		value = events.realize_known(output_sockets['Structure'])
		if value is not None:
			return value
		return FS.FlowPending

	####################
	# - FlowKind.Func
	####################
	@events.computes_output_socket(
		'Structure',
		kind=FK.Func,
		# Loaded
		inscks_kinds={
			'Medium': FK.Func,
			'Center': FK.Func,
			'Radius': FK.Func,
		},
		scale_input_sockets={
			'Center': ct.UNITS_TIDY3D,
			'Radius': ct.UNITS_TIDY3D,
		},
	)
	def compute_func(self, input_sockets) -> td.Box:
		"""Compute a function, producing a box structure from the input parameters."""
		center = input_sockets['Center']
		radius = input_sockets['Radius']
		medium = input_sockets['Medium']

		return (center | radius | medium).compose_within(
			lambda els: td.Structure(
				geometry=td.Sphere(
					center=els[0].flatten().tolist(),
					radius=els[1],
				),
				medium=els[2],
			),
		)

	####################
	# - FlowKind.Params
	####################
	@events.computes_output_socket(
		'Structure',
		kind=FK.Params,
		# Loaded
		inscks_kinds={
			'Medium': FK.Params,
			'Center': FK.Params,
			'Radius': FK.Params,
		},
	)
	def compute_params(self, input_sockets) -> td.Box:
		"""Aggregate the function parameters needed by the sphere."""
		center = input_sockets['Center']
		radius = input_sockets['Radius']
		medium = input_sockets['Medium']

		return center | radius | medium

	####################
	# - Preview
	####################
	@events.computes_output_socket(
		'Structure',
		kind=FK.Previews,
		# Loaded
		props={'sim_node_name'},
	)
	def compute_previews(self, props):
		"""Mark the sphere structure as participating in the preview."""
		return ct.PreviewsFlow(bl_object_names={props['sim_node_name']})

	@events.on_value_changed(
		# Trigger
		socket_name={
			'Center': {FK.Func, FK.Params},
			'Size': {FK.Func, FK.Params},
		},
		run_on_init=True,
		# Loaded
		managed_objs={'modifier'},
		inscks_kinds={
			'Center': {FK.Func, FK.Params},
			'Radius': {FK.Func, FK.Params},
		},
		scale_input_sockets={
			'Center': ct.UNITS_BLENDER,
			'Radius': ct.UNITS_BLENDER,
		},
	)
	def on_previewable_changed(self, managed_objs, input_sockets):
		"""Push changes in the inputs to the center / size."""
		center = events.realize_preview(input_sockets['Center'])
		radius = events.realize_preview(input_sockets['Radius'])

		managed_objs['modifier'].bl_modifier(
			'NODES',
			{
				'node_group': import_geonodes(GeoNodes.StructurePrimitiveSphere),
				'inputs': {
					'Radius': radius,
				},
			},
			location=center,
		)


####################
# - Blender Registration
####################
BL_REGISTER = [
	SphereStructureNode,
]
BL_NODES = {
	ct.NodeType.SphereStructure: (ct.NodeCategory.MAXWELLSIM_STRUCTURES_PRIMITIVES)
}
