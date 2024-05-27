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

from blender_maxwell.utils import extra_sympy_units as spux
from blender_maxwell.utils import logger

from ... import bl_socket_map, managed_objs, sockets
from ... import contracts as ct
from .. import base, events

log = logger.get(__name__)


class GeoNodesStructureNode(base.MaxwellSimNode):
	node_type = ct.NodeType.GeoNodesStructure
	bl_label = 'GeoNodes Structure'
	use_sim_node_name = True

	####################
	# - Sockets
	####################
	input_sockets: typ.ClassVar = {
		'GeoNodes': sockets.BlenderGeoNodesSocketDef(),
		'Medium': sockets.MaxwellMediumSocketDef(),
		'Center': sockets.ExprSocketDef(
			size=spux.NumberSize1D.Vec3,
			default_unit=spu.micrometer,
			default_value=sp.Matrix([0, 0, 0]),
		),
	}
	output_sockets: typ.ClassVar = {
		'Structure': sockets.MaxwellStructureSocketDef(),
	}

	managed_obj_types: typ.ClassVar = {
		'modifier': managed_objs.ManagedBLModifier,
	}

	####################
	# - Outputs
	####################
	@events.computes_output_socket(
		'Structure',
		input_sockets={'Medium'},
		managed_objs={'modifier'},
	)
	def compute_structure(
		self,
		input_sockets,
		managed_objs,
	) -> td.Structure:
		"""Computes a triangle-mesh based Tidy3D structure, by manually copying mesh data from Blender to a `td.TriangleMesh`."""
		## TODO: mesh_as_arrays might not take the Center into account.
		## - Alternatively, Tidy3D might have a way to transform?
		mesh_as_arrays = managed_objs['modifier'].mesh_as_arrays
		return td.Structure(
			geometry=td.TriangleMesh.from_vertices_faces(
				mesh_as_arrays['verts'],
				mesh_as_arrays['faces'],
			),
			medium=input_sockets['Medium'],
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
		socket_name={'GeoNodes'},
		# Loaded
		managed_objs={'modifier'},
		input_sockets={'GeoNodes'},
	)
	def on_input_changed(
		self,
		managed_objs,
		input_sockets,
	) -> None:
		"""Declares new loose input sockets in response to a new GeoNodes tree (if any)."""
		geonodes = input_sockets['GeoNodes']
		has_geonodes = not ct.FlowSignal.check(geonodes) and geonodes is not None

		if has_geonodes:
			# Fill the Loose Input Sockets
			## -> The SocketDefs contain the default values from the interface.
			log.info(
				'Initializing GeoNodes Structure Node "%s" from GeoNodes Group "%s"',
				self.bl_label,
				str(geonodes),
			)
			self.loose_input_sockets = bl_socket_map.sockets_from_geonodes(geonodes)

			## -> The loose socket creation triggers 'on_input_socket_changed'

		elif self.loose_input_sockets:
			self.loose_input_sockets = {}
			managed_objs['modifier'].free()

	####################
	# - Events: Preview
	####################
	@events.computes_output_socket(
		'Structure',
		kind=ct.FlowKind.Previews,
		# Loaded
		props={'sim_node_name'},
	)
	def compute_previews(self, props):
		return ct.PreviewsFlow(bl_object_names={props['sim_node_name']})

	@events.on_value_changed(
		# Trigger
		socket_name={'Center', 'GeoNodes'},  ## MUST run after on_input_changed
		any_loose_input_socket=True,
		# Loaded
		managed_objs={'modifier'},
		input_sockets={'Center', 'GeoNodes'},
		all_loose_input_sockets=True,
		unit_systems={'BlenderUnits': ct.UNITS_BLENDER},
		scale_input_sockets={'Center': 'BlenderUnits'},
	)
	def on_input_socket_changed(
		self, managed_objs, input_sockets, loose_input_sockets, unit_systems
	) -> None:
		"""Pushes any change in GeoNodes-bound input sockets to the GeoNodes modifier.

		Warnings:
			MUST be placed lower than `on_input_changed`, so it runs afterwards whenever the `GeoNodes` tree is changed.

		Also pushes the `Center:Value` socket to govern the object's center in 3D space.
		"""
		geonodes = input_sockets['GeoNodes']
		has_geonodes = not ct.FlowSignal.check(geonodes) and geonodes is not None

		if has_geonodes:
			# Push Loose Input Values to GeoNodes Modifier
			managed_objs['modifier'].bl_modifier(
				'NODES',
				{
					'node_group': geonodes,
					'inputs': loose_input_sockets,
					'unit_system': unit_systems['BlenderUnits'],
				},
				location=input_sockets['Center'],
			)


####################
# - Blender Registration
####################
BL_REGISTER = [
	GeoNodesStructureNode,
]
BL_NODES = {ct.NodeType.GeoNodesStructure: (ct.NodeCategory.MAXWELLSIM_STRUCTURES)}
