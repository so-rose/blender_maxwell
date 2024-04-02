import bmesh
import bpy
import numpy as np
import tidy3d as td

from ... import contracts, sockets
from .. import base


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
	@base.computes_output_socket('structure')
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
		faces = np.array(
			[[vert for vert in poly.vertices] for poly in mesh.polygons]
		)

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
	contracts.NodeType.ObjectStructure: (
		contracts.NodeCategory.MAXWELLSIM_STRUCTURES
	)
}
