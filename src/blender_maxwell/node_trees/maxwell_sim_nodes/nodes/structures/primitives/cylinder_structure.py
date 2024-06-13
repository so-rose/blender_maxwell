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

"""Implements `CylinderStructureNode`."""

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


class CylinderStructureNode(base.MaxwellSimNode):
	"""A generic cylinder structure with configurable radius and height."""

	node_type = ct.NodeType.CylinderStructure
	bl_label = 'Cylinder Structure'
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
		'Height': sockets.ExprSocketDef(
			default_unit=spu.nanometer,
			default_value=500,
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
		output_sockets={'Structure'},
		output_socket_kinds={'Structure': {FK.Func, FK.Params}},
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
			'Center': FK.Func,
			'Radius': FK.Func,
			'Height': FK.Func,
			'Medium': FK.Func,
		},
		scale_input_sockets={
			'Center': ct.UNITS_TIDY3D,
			'Radius': ct.UNITS_TIDY3D,
			'Height': ct.UNITS_TIDY3D,
		},
	)
	def compute_func(self, input_sockets) -> td.Box:
		"""Compute a single cylinder structure object, given that all inputs are non-symbolic."""
		center = input_sockets['Center']
		radius = input_sockets['Radius']
		height = input_sockets['Height']
		medium = input_sockets['Medium']

		return (center | radius | height | medium).compose_within(
			lambda els: td.Structure(
				geometry=td.Cylinder(
					center=els[0].flatten().tolist(),
					radius=els[1],
					length=els[2],
				),
				medium=els[3],
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
			'Center': FK.Params,
			'Radius': FK.Params,
			'Height': FK.Params,
			'Medium': FK.Params,
		},
	)
	def compute_params(self, input_sockets) -> td.Box:
		"""Aggregate the function parameters needed by the cylinder."""
		center = input_sockets['Center']
		radius = input_sockets['Radius']
		height = input_sockets['Height']
		medium = input_sockets['Medium']

		return center | radius | height | medium

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
		"""Mark the cylinder structure as participating in the preview."""
		return ct.PreviewsFlow(bl_object_names={props['sim_node_name']})

	@events.on_value_changed(
		# Trigger
		socket_name={'Center', 'Radius', 'Medium', 'Height'},
		run_on_init=True,
		# Loaded
		managed_objs={'modifier'},
		inscks_kinds={
			'Center': {FK.Func, FK.Params},
			'Radius': {FK.Func, FK.Params},
			'Medium': {FK.Func, FK.Params},
			'Height': {FK.Func, FK.Params},
		},
		scale_input_sockets={
			'Center': ct.UNITS_BLENDER,
			'Radius': ct.UNITS_BLENDER,
			'Height': ct.UNITS_BLENDER,
		},
	)
	def on_previewable_changed(self, managed_objs, input_sockets) -> None:
		"""Push changes in the inputs to the center / size."""
		center = events.realize_preview(input_sockets['Center'])
		radius = events.realize_preview(input_sockets['Radius'])
		height = events.realize_preview(input_sockets['Height'])

		# Push Loose Input Values to GeoNodes Modifier
		managed_objs['modifier'].bl_modifier(
			'NODES',
			{
				'node_group': import_geonodes(GeoNodes.StructurePrimitiveCylinder),
				'inputs': {
					'Radius': radius,
					'Height': height,
				},
				'unit_system': ct.UNITS_BLENDER,
			},
			location=center,
		)


####################
# - Blender Registration
####################
BL_REGISTER = [
	CylinderStructureNode,
]
BL_NODES = {
	ct.NodeType.CylinderStructure: (ct.NodeCategory.MAXWELLSIM_STRUCTURES_PRIMITIVES)
}
