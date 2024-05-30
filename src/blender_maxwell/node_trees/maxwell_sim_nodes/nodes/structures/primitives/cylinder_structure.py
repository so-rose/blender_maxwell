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

import sympy as sp
import sympy.physics.units as spu
import tidy3d as td

from blender_maxwell.assets.geonodes import GeoNodes, import_geonodes
from blender_maxwell.utils import extra_sympy_units as spux
from blender_maxwell.utils import logger

from .... import contracts as ct
from .... import managed_objs, sockets
from ... import base, events

log = logger.get(__name__)


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
			default_value=sp.Matrix([0, 0, 0]),
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
		'Structure': sockets.MaxwellStructureSocketDef(active_kind=ct.FlowKind.Func),
	}

	managed_obj_types: typ.ClassVar = {
		'modifier': managed_objs.ManagedBLModifier,
	}

	####################
	# - FlowKind.Value
	####################
	@events.computes_output_socket(
		'Structure',
		kind=ct.FlowKind.Value,
		# Loaded
		output_sockets={'Structure'},
		output_socket_kinds={'Structure': {ct.FlowKind.Func, ct.FlowKind.Params}},
	)
	def compute_value(self, output_sockets) -> ct.ParamsFlow | ct.FlowSignal:
		"""Compute the particular value of the simulation domain from strictly non-symbolic inputs."""
		output_func = output_sockets['Structure'][ct.FlowKind.Func]
		output_params = output_sockets['Structure'][ct.FlowKind.Params]

		has_output_func = not ct.FlowSignal.check(output_func)
		has_output_params = not ct.FlowSignal.check(output_params)

		if has_output_func and has_output_params and not output_params.symbols:
			return output_func.realize(output_params, disallow_jax=True)
		return ct.FlowSignal.FlowPending

	####################
	# - FlowKind.Func
	####################
	@events.computes_output_socket(
		'Structure',
		kind=ct.FlowKind.Func,
		# Loaded
		input_sockets={'Center', 'Radius', 'Medium', 'Height'},
		input_socket_kinds={
			'Center': ct.FlowKind.Func,
			'Radius': ct.FlowKind.Func,
			'Height': ct.FlowKind.Func,
			'Medium': ct.FlowKind.Func,
		},
	)
	def compute_func(self, input_sockets) -> td.Box:
		"""Compute a single cylinder structure object, given that all inputs are non-symbolic."""
		center = input_sockets['Center']
		radius = input_sockets['Radius']
		height = input_sockets['Height']
		medium = input_sockets['Medium']

		has_center = not ct.FlowSignal.check(center)
		has_radius = not ct.FlowSignal.check(radius)
		has_height = not ct.FlowSignal.check(height)
		has_medium = not ct.FlowSignal.check(medium)

		if has_center and has_radius and has_height and has_medium:
			return (
				center.scale_to_unit_system(ct.UNITS_TIDY3D)
				| radius.scale_to_unit_system(ct.UNITS_TIDY3D)
				| height.scale_to_unit_system(ct.UNITS_TIDY3D)
				| medium
			).compose_within(
				lambda els: td.Structure(
					geometry=td.Cylinder(
						center=els[0].flatten().tolist(),
						radius=els[1],
						length=els[2],
					),
					medium=els[3],
				)
			)
		return ct.FlowSignal.FlowPending

	####################
	# - FlowKind.Params
	####################
	@events.computes_output_socket(
		'Structure',
		kind=ct.FlowKind.Params,
		# Loaded
		input_sockets={'Center', 'Radius', 'Medium', 'Height'},
		input_socket_kinds={
			'Center': ct.FlowKind.Params,
			'Radius': ct.FlowKind.Params,
			'Height': ct.FlowKind.Params,
			'Medium': ct.FlowKind.Params,
		},
	)
	def compute_params(self, input_sockets) -> td.Box:
		center = input_sockets['Center']
		radius = input_sockets['Radius']
		height = input_sockets['Height']
		medium = input_sockets['Medium']

		has_center = not ct.FlowSignal.check(center)
		has_radius = not ct.FlowSignal.check(radius)
		has_height = not ct.FlowSignal.check(height)
		has_medium = not ct.FlowSignal.check(medium)

		if has_center and has_radius and has_height and has_medium:
			return center | radius | height | medium
		return ct.FlowSignal.FlowPending

	####################
	# - Preview
	####################
	@events.computes_output_socket(
		'Structure',
		kind=ct.FlowKind.Previews,
		# Loaded
		props={'sim_node_name'},
		output_sockets={'Structure'},
		output_socket_kinds={'Structure': ct.FlowKind.Params},
	)
	def compute_previews(self, props, output_sockets):
		output_params = output_sockets['Structure']
		has_output_params = not ct.FlowSignal.check(output_params)

		if has_output_params and not output_params.symbols:
			return ct.PreviewsFlow(bl_object_names={props['sim_node_name']})
		return ct.PreviewsFlow()

	@events.on_value_changed(
		# Trigger
		socket_name={'Center', 'Radius', 'Medium', 'Height'},
		run_on_init=True,
		# Loaded
		input_sockets={'Center', 'Radius', 'Medium', 'Height'},
		managed_objs={'modifier'},
		output_sockets={'Structure'},
		output_socket_kinds={'Structure': ct.FlowKind.Params},
	)
	def on_previewable_changed(self, managed_objs, input_sockets, output_sockets):
		center = input_sockets['Center']
		radius = input_sockets['Radius']
		height = input_sockets['Height']
		output_params = output_sockets['Structure']

		has_center = not ct.FlowSignal.check(center)
		has_radius = not ct.FlowSignal.check(radius)
		has_height = not ct.FlowSignal.check(height)
		has_output_params = not ct.FlowSignal.check(output_params)

		if (
			has_center
			and has_radius
			and has_height
			and has_output_params
			and not output_params.symbols
		):
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
				location=spux.scale_to_unit_system(center, ct.UNITS_BLENDER),
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
