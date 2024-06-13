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

"""Implements `GeoNodesStructureNode`."""

import functools
import typing as typ

import sympy as sp
import sympy.physics.units as spu
import tidy3d as td

from blender_maxwell.utils import logger
from blender_maxwell.utils import sympy_extra as spux

from ... import bl_socket_map, managed_objs, sockets
from ... import contracts as ct
from .. import base, events

log = logger.get(__name__)

FK = ct.FlowKind
FS = ct.FlowSignal
MT = spux.MathType


class GeoNodesStructureNode(base.MaxwellSimNode):
	"""A generic mesh structure defined by an arbitrary Geometry Nodes tree."""

	node_type = ct.NodeType.GeoNodesStructure
	bl_label = 'GeoNodes Structure'
	use_sim_node_name = True

	####################
	# - Sockets
	####################
	input_sockets: typ.ClassVar = {
		'GeoNodes': sockets.BlenderGeoNodesSocketDef(),
		'Medium': sockets.MaxwellMediumSocketDef(active_kind=FK.Func),
		'Center': sockets.ExprSocketDef(
			size=spux.NumberSize1D.Vec3,
			default_unit=spu.micrometer,
			default_value=sp.ImmutableMatrix([0, 0, 0]),
		),
	}
	output_sockets: typ.ClassVar = {
		'Structure': sockets.MaxwellStructureSocketDef(active_kind=FK.Func),
	}

	managed_obj_types: typ.ClassVar = {
		'preview_mesh': managed_objs.ManagedBLModifier,
		'mesh': managed_objs.ManagedBLModifier,
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
		props={'sim_node_name'},
		managed_objs={'mesh'},
		inscks_kinds={
			'GeoNodes': FK.Value,
			'Medium': FK.Func,
			'Center': FK.Func,
		},
		scale_input_sockets={
			'Center': ct.UNITS_TIDY3D,
		},
		all_loose_input_sockets=True,
		loose_input_sockets_kind=FK.Func,
		scale_loose_input_sockets=ct.UNITS_BLENDER,
	)
	def compute_func(
		self,
		managed_objs,
		props,
		input_sockets,
		loose_input_sockets,
	) -> td.Structure:
		"""Lazily computes a triangle-mesh based Tidy3D structure, by manually copying mesh data from Blender to a `td.TriangleMesh`."""
		## TODO: mesh_as_arrays might not take the Center into account.
		## - Alternatively, Tidy3D might have a way to transform?

		mesh = managed_objs['mesh']
		geonodes = input_sockets['GeoNodes']
		medium = input_sockets['Medium']
		center = input_sockets['Center']

		sim_node_name = props['sim_node_name']
		gn_inputs = list(loose_input_sockets.keys())

		def verts_faces(els: tuple[typ.Any]) -> dict[str, typ.Any]:
			# Push Realized Values to Managed Mesh
			mesh.bl_modifier(
				'NODES',
				{
					'node_group': geonodes,
					'inputs': dict(zip(gn_inputs, els[2:], strict=True)),
				},
				location=els[1],
			)

			# Extract Vert/Face Data
			mesh_as_arrays = mesh.mesh_as_arrays
			return (mesh_as_arrays['verts'], mesh_as_arrays['faces'])

		loose_sck_values = functools.reduce(
			lambda a, b: a | b, loose_input_sockets.values()
		)
		return (medium | center | loose_sck_values).compose_within(
			lambda els: td.Structure(
				name=sim_node_name,
				geometry=td.TriangleMesh.from_vertices_faces(
					*verts_faces(els),
				),
				medium=els[0],
			)
		)

	####################
	# - UI
	####################
	def draw_label(self) -> None:
		"""Show the extracted data (if any) in the node's header label.

		Notes:
			Called by Blender to determine the text to place in the node's header.
		"""
		geonodes = self._compute_input('GeoNodes')
		if geonodes is None:
			return self.bl_label

		return f'Structure: {self.sim_node_name}'

	####################
	# - Events: Swapped GN Node Tree
	####################
	@events.on_value_changed(
		socket_name={
			'GeoNodes': FK.Value,
		},
		# Loaded
		inscks_kinds={
			'GeoNodes': FK.Value,
		},
	)
	def on_input_changed(
		self,
		input_sockets,
	) -> None:
		"""Synchronizes the GeoNodes tree input sockets to the loose input sockets of this node.

		Utilizes `bl_socket_map.sockets_from_geonodes` to generate, primarily, `ExprSocketDef`s for use with `self.loose_input_sockets.
		"""
		geonodes = input_sockets['GeoNodes']
		if geonodes is not None:
			# Fill the Loose Input Sockets
			## -> The SocketDefs contain the default values from the interface.
			log.debug(
				'Initializing GeoNodes Structure Node "%s" from GeoNodes Group "%s"',
				self.bl_label,
				str(geonodes),
			)
			self.loose_input_sockets = bl_socket_map.sockets_from_geonodes(geonodes)

		elif self.loose_input_sockets:
			self.loose_input_sockets = {}

	####################
	# - Events: Preview
	####################
	@events.computes_output_socket(
		'Structure',
		kind=FK.Previews,
		# Loaded
		props={'sim_node_name'},
	)
	def compute_previews(self, props):
		"""Mark the box structure as participating in the preview."""
		sim_node_name = props['sim_node_name']
		return ct.PreviewsFlow(bl_object_names={sim_node_name + '_1'})

	@events.on_value_changed(
		# Trigger
		socket_name={
			'GeoNodes': FK.Value,
			'Center': {FK.Func, FK.Params},
		},
		any_loose_input_socket=True,
		run_on_init=True,
		# Loaded
		managed_objs={'preview_mesh'},
		inscks_kinds={
			'GeoNodes': FK.Value,
			'Center': {FK.Func, FK.Params},
		},
		scale_input_sockets={
			'Center': ct.UNITS_BLENDER,
		},
		all_loose_input_sockets=True,
		loose_input_sockets_kind={FK.Func, FK.Params},
		scale_loose_input_sockets=ct.UNITS_BLENDER,
	)
	def on_input_socket_changed(
		self, managed_objs, input_sockets, loose_input_sockets
	) -> None:
		"""Pushes any change in GeoNodes-bound input sockets to the GeoNodes modifier.

		Warnings:
			MUST be placed lower than `on_input_changed`, so it runs afterwards whenever the `GeoNodes` tree is changed.

		Also pushes the `Center:Value` socket to govern the object's center in 3D space.
		"""
		geonodes = input_sockets['GeoNodes']
		center = events.realize_preview(input_sockets['Center'])

		managed_objs['modifier'].bl_modifier(
			'NODES',
			{
				'node_group': geonodes,
				'inputs': loose_input_sockets,
			},
			location=center,
		)


####################
# - Blender Registration
####################
BL_REGISTER = [
	GeoNodesStructureNode,
]
BL_NODES = {ct.NodeType.GeoNodesStructure: (ct.NodeCategory.MAXWELLSIM_STRUCTURES)}
