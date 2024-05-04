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

import bmesh
import bpy
import numpy as np
import tidy3d as td

from ... import contracts, sockets
from .. import base, events


class ObjectStructureNode(base.MaxwellSimTreeNode):
	node_type = contracts.NodeType.ObjectStructure
	bl_label = 'Object Structure'
	# bl_icon = ...

	####################
	# - Sockets
	####################
	input_sockets = {
		'medium': sockets.MaxwellMediumSocketDef(
			label='Medium',
		),
		'object': sockets.BlenderObjectSocketDef(
			label='Object',
		),
	}
	output_sockets = {
		'structure': sockets.MaxwellStructureSocketDef(
			label='Structure',
		),
	}

	####################
	# - Output Socket Computation
	####################
	@events.computes_output_socket('structure')
	def compute_structure(self: contracts.NodeTypeProtocol) -> td.Structure:
		# Extract the Blender Object
		bl_object = self.compute_input('object')

		# Ensure Updated Geometry
		bpy.context.view_layer.update()

		# Triangulate Object Mesh
		bmesh_mesh = bmesh.new()
		bmesh_mesh.from_object(
			bl_object,
			bpy.context.evaluated_depsgraph_get(),
		)
		bmesh.ops.triangulate(bmesh_mesh, faces=bmesh_mesh.faces)

		mesh = bpy.data.meshes.new(name='TriangulatedMesh')
		bmesh_mesh.to_mesh(mesh)
		bmesh_mesh.free()

		# Extract Vertices and Faces
		vertices = np.array([vert.co for vert in mesh.vertices])
		faces = np.array([[vert for vert in poly.vertices] for poly in mesh.polygons])

		# Remove Temporary Mesh
		bpy.data.meshes.remove(mesh)

		return td.Structure(
			geometry=td.TriangleMesh.from_vertices_faces(vertices, faces),
			medium=self.compute_input('medium'),
		)


####################
# - Blender Registration
####################
BL_REGISTER = [
	ObjectStructureNode,
]
BL_NODES = {
	contracts.NodeType.ObjectStructure: (contracts.NodeCategory.MAXWELLSIM_STRUCTURES)
}
