import tidy3d as td
import numpy as np
import sympy as sp
import sympy.physics.units as spu

import bpy
import bmesh

from ... import contracts
from ... import sockets
from .. import base

GEONODES_MODIFIER_NAME = "BLMaxwell_GeoNodes"

class GeoNodesStructureNode(base.MaxwellSimTreeNode):
	node_type = contracts.NodeType.GeoNodesStructure
	bl_label = "GeoNodes Structure"
	#bl_icon = ...
	
	####################
	# - Sockets
	####################
	input_sockets = {
		"medium": sockets.MaxwellMediumSocketDef(
			label="Medium",
		),
		"object": sockets.BlenderObjectSocketDef(
			label="Object",
		),
		"geo_nodes": sockets.BlenderGeoNodesSocketDef(
			label="GeoNodes",
		),
	}
	output_sockets = {
		"structure": sockets.MaxwellStructureSocketDef(
			label="Structure",
		),
	}
	
	####################
	# - Output Socket Computation
	####################
	@base.computes_output_socket("structure")
	def compute_simulation(self: contracts.NodeTypeProtocol) -> td.TriangleMesh:
		# Extract the Blender Object
		bl_object = self.compute_input("object")
		
		# Ensure Updated Geometry
		bpy.context.view_layer.update()
		
		# Triangulate Object Mesh
		bmesh_mesh = bmesh.new()
		bmesh_mesh.from_mesh(bl_object.data)
		bmesh.ops.triangulate(bmesh_mesh, faces=bmesh_mesh.faces)
		
		mesh = bpy.data.meshes.new(name="TriangulatedMesh")
		bmesh_mesh.to_mesh(mesh)
		bmesh_mesh.free()
		
		# Extract Vertices and Faces
		vertices = np.array([vert.co for vert in mesh.vertices])
		faces = np.array([
			[vert for vert in poly.vertices]
			for poly in mesh.polygons
		])
		
		# Remove Temporary Mesh
		bpy.data.meshes.remove(mesh)
		
		return td.Structure(
			geometry=td.TriangleMesh.from_vertices_faces(vertices, faces),
			medium=self.compute_input("medium")
		)
	
	####################
	# - Update Function
	####################
	def update_cb(self) -> None:
		bl_object = self.compute_input("object")
		if bl_object is None: return
		
		geo_nodes = self.compute_input("geo_nodes")
		if geo_nodes is None: return
		
		bl_modifier = bl_object.modifiers.get(GEONODES_MODIFIER_NAME)
		if bl_modifier is None: return
		
		# Set GeoNodes Modifier Attributes
		for idx, interface_item in enumerate(
			geo_nodes.interface.items_tree.values()
		):
			if idx == 0: continue  ## Always-on "Geometry" Input (from Object)
			
			bl_socket = self.inputs[
				interface_item.name
			]
			if bl_socket.is_linked:
				linked_bl_socket = bl_socket.links[0].from_socket
				linked_bl_node = bl_socket.links[0].from_node
				val = linked_bl_node.compute_output(
					linked_bl_node.g_output_socket_name(
						linked_bl_socket.name
					)
				)  ## What a bunch of spaghetti
			else:
				val = self.inputs[
					interface_item.name
				].default_value
			
			# Conservatively Set Differing Values
			if bl_modifier[interface_item.identifier] != val:
				bl_modifier[interface_item.identifier] = val
			
		# Update DepGraph
		bl_object.data.update()
		
	def update_sockets_from_geonodes(self) -> None:
		# Remove All "Loose" Sockets
		socket_labels = {
			socket_def.label
			for socket_def in self.input_sockets.values()
		} | {
			socket_def.label
			for socket_set_name, socket_set in self.input_socket_sets.items()
			for socket_name, socket_def in socket_set.items()
		}
		bl_sockets_to_remove = {
			bl_socket
			for bl_socket_name, bl_socket in self.inputs.items()
			if bl_socket_name not in socket_labels
		}
		
		for bl_socket in bl_sockets_to_remove:
			self.inputs.remove(bl_socket)
		
		# Query for Blender Object / Geo Nodes
		bl_object = self.compute_input("object")
		if bl_object is None: return
		## TODO: Make object? Gray out geonodes if object not defined?
		
		geo_nodes = self.compute_input("geo_nodes")
		if geo_nodes is None: return
		
		
		# Add Non-Static Sockets from GeoNodes
		for bl_socket_name, bl_socket in geo_nodes.interface.items_tree.items():
			# For now, don't allow Geometry inputs.
			if bl_socket.socket_type == "NodeSocketGeometry": continue
			
			self.inputs.new(
				contracts.BLNodeSocket_to_SocketType[bl_socket.socket_type],
				bl_socket_name,
			)
		
		# Create New GeoNodes Modifier
		if GEONODES_MODIFIER_NAME not in bl_object.modifiers:
			modifier = bl_object.modifiers.new(
				name=GEONODES_MODIFIER_NAME,
				type="NODES",
			)
			modifier.node_group = geo_nodes
		
		self.update()



####################
# - Blender Registration
####################
BL_REGISTER = [
	GeoNodesStructureNode,
]
BL_NODES = {
	contracts.NodeType.GeoNodesStructure: (
		contracts.NodeCategory.MAXWELLSIM_STRUCTURES
	)
}
