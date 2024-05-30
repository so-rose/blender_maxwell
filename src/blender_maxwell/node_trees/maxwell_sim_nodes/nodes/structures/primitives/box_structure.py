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
import sympy.physics.units as spu
import tidy3d as td
import tidy3d.plugins.adjoint as tdadj

from blender_maxwell.assets.geonodes import GeoNodes, import_geonodes
from blender_maxwell.utils import bl_cache, logger
from blender_maxwell.utils import extra_sympy_units as spux

from .... import contracts as ct
from .... import managed_objs, sockets
from ... import base, events

log = logger.get(__name__)


class BoxStructureNode(base.MaxwellSimNode):
	"""A generic, differentiable box structure with configurable size and center."""

	node_type = ct.NodeType.BoxStructure
	bl_label = 'Box Structure'
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
		'Size': sockets.ExprSocketDef(
			size=spux.NumberSize1D.Vec3,
			default_unit=spu.nanometer,
			default_value=sp.Matrix([500, 500, 500]),
			abs_min=0.001,
		),
	}
	output_sockets: typ.ClassVar = {
		'Structure': sockets.MaxwellStructureSocketDef(active_kind=ct.FlowKind.Func),
	}

	managed_obj_types: typ.ClassVar = {
		'modifier': managed_objs.ManagedBLModifier,
	}

	####################
	# - Properties
	####################
	differentiable: bool = bl_cache.BLField(False)

	####################
	# - UI
	####################
	def draw_props(self, _: bpy.types.Context, layout: bpy.types.UILayout):
		layout.prop(
			self,
			self.blfields['differentiable'],
			text='Differentiable',
			toggle=True,
		)

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
		props={'differentiable'},
		input_sockets={'Medium', 'Center', 'Size'},
		input_socket_kinds={
			'Medium': ct.FlowKind.Func,
			'Center': ct.FlowKind.Func,
			'Size': ct.FlowKind.Func,
		},
	)
	def compute_structure_func(self, props, input_sockets) -> td.Box:
		"""Compute a possibly-differentiable function, producing a box structure from the input parameters."""
		center = input_sockets['Center']
		size = input_sockets['Size']
		medium = input_sockets['Medium']

		has_center = not ct.FlowSignal.check(center)
		has_size = not ct.FlowSignal.check(size)
		has_medium = not ct.FlowSignal.check(medium)

		if has_center and has_size and has_medium:
			differentiable = props['differentiable']
			if differentiable:
				return (
					center.scale_to_unit_system(ct.UNITS_TIDY3D)
					| size.scale_to_unit_system(ct.UNITS_TIDY3D)
					| medium
				).compose_within(
					lambda els: tdadj.JaxStructure(
						geometry=tdadj.JaxBox(
							center_jax=tuple(els[0].flatten()),
							size_jax=tuple(els[1].flatten()),
						),
						medium=els[2],
					),
					supports_jax=True,
				)
			return (
				center.scale_to_unit_system(ct.UNITS_TIDY3D)
				| size.scale_to_unit_system(ct.UNITS_TIDY3D)
				| medium
			).compose_within(
				lambda els: td.Structure(
					geometry=td.Box(
						center=els[0].flatten().tolist(),
						size=els[1].flatten().tolist(),
					),
					medium=els[2],
				),
				supports_jax=False,
			)
		return ct.FlowSignal.FlowPending

	####################
	# - FlowKind.Params
	####################
	@events.computes_output_socket(
		'Structure',
		kind=ct.FlowKind.Params,
		# Loaded
		input_sockets={'Medium', 'Center', 'Size'},
		input_socket_kinds={
			'Medium': ct.FlowKind.Params,
			'Center': ct.FlowKind.Params,
			'Size': ct.FlowKind.Params,
		},
	)
	def compute_params(self, input_sockets) -> td.Box:
		center = input_sockets['Center']
		size = input_sockets['Size']
		medium = input_sockets['Medium']

		has_center = not ct.FlowSignal.check(center)
		has_size = not ct.FlowSignal.check(size)
		has_medium = not ct.FlowSignal.check(medium)

		if has_center and has_size and has_medium:
			return center | size | medium
		return ct.FlowSignal.FlowPending

	####################
	# - Events: Preview
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
		"""Mark the managed preview object when recursively linked to a viewer."""
		output_params = output_sockets['Structure']
		has_output_params = not ct.FlowSignal.check(output_params)

		if has_output_params and not output_params.symbols:
			return ct.PreviewsFlow(bl_object_names={props['sim_node_name']})
		return ct.PreviewsFlow()

	@events.on_value_changed(
		# Trigger
		socket_name={'Center', 'Size'},
		run_on_init=True,
		# Loaded
		input_sockets={'Center', 'Size'},
		managed_objs={'modifier'},
		output_sockets={'Structure'},
		output_socket_kinds={'Structure': ct.FlowKind.Params},
	)
	def on_previewable_changed(self, managed_objs, input_sockets, output_sockets):
		center = input_sockets['Center']
		size = input_sockets['Size']
		output_params = output_sockets['Structure']

		has_center = not ct.FlowSignal.check(center)
		has_size = not ct.FlowSignal.check(size)
		has_output_params = not ct.FlowSignal.check(output_params)

		if has_center and has_size and has_output_params and not output_params.symbols:
			# Push Loose Input Values to GeoNodes Modifier
			managed_objs['modifier'].bl_modifier(
				'NODES',
				{
					'node_group': import_geonodes(GeoNodes.StructurePrimitiveBox),
					'unit_system': ct.UNITS_BLENDER,
					'inputs': {
						'Size': size,
					},
				},
				location=spux.scale_to_unit_system(center, ct.UNITS_BLENDER),
			)


####################
# - Blender Registration
####################
BL_REGISTER = [
	BoxStructureNode,
]
BL_NODES = {
	ct.NodeType.BoxStructure: (ct.NodeCategory.MAXWELLSIM_STRUCTURES_PRIMITIVES)
}
