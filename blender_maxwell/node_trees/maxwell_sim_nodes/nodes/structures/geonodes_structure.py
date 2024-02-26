import tidy3d as td
import numpy as np
import sympy as sp
import sympy.physics.units as spu

import bpy
from bpy_types import bpy_types
import bmesh

from ... import contracts
from ... import sockets
from .. import base

GEONODES_MODIFIER_NAME = "BLMaxwell_GeoNodes"

# Monkey-Patch Sympy Types
## TODO: This needs to be a more generic thing, this isn't the only place we're setting blender interface values.
def parse_scalar(scalar):
	if isinstance(scalar, sp.Integer):
		return int(scalar)
	elif isinstance(scalar, sp.Float):
		return float(scalar)
	elif isinstance(scalar, sp.Rational):
		return float(scalar)
	elif isinstance(scalar, sp.Expr):
		return float(scalar.n())
	
	return scalar

def parse_bl_to_sp(scalar):
	if isinstance(scalar, bpy_types.bpy_prop_array):
		return sp.Matrix(tuple(scalar))
	
	return scalar

class GeoNodesStructureNode(base.MaxwellSimTreeNode):
	node_type = contracts.NodeType.GeoNodesStructure
	bl_label = "GeoNodes Structure"
	#bl_icon = ...
	
	####################
	# - Sockets
	####################
	input_sockets = {
		"preview_target": sockets.BlenderPreviewTargetSocketDef(
			label="Preview Target",
		),
		"blender_unit_system": sockets.PhysicalUnitSystemSocketDef(
			label="Blender Units",
		),
		"medium": sockets.MaxwellMediumSocketDef(
			label="Medium",
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
		bmesh_mesh.from_object(
			bl_object,
			bpy.context.evaluated_depsgraph_get(),
		)
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
	def free(self) -> None:
		bl_socket = self.g_input_bl_socket("preview_target")
		
		bl_socket.free()
		
	def update_cb(self) -> None:
		bl_object = self.compute_input("preview_target")
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
			
			# Retrieve Input Socket
			bl_socket = self.inputs[
				interface_item.name
			]
			
			# Retrieve Linked/Unlinked Input Socket Value
			if bl_socket.is_linked:
				linked_bl_socket = bl_socket.links[0].from_socket
				linked_bl_node = bl_socket.links[0].from_node
				val = linked_bl_node.compute_output(
					linked_bl_node.g_output_socket_name(
						linked_bl_socket.name
					)
				)  ## What a bunch of spaghetti
			else:
				val = bl_socket.default_value
			
			# Retrieve Unit-System Corrected Modifier Value
			bl_unit_system = self.compute_input("blender_unit_system")
			
			socket_type = contracts.SocketType[
				bl_socket.bl_idname.removesuffix("SocketType")
			]
			if socket_type in bl_unit_system:
				unitless_val = spu.convert_to(
					val,
					bl_unit_system[socket_type],
				) / bl_unit_system[socket_type]
			else:
				unitless_val = val
			
			if isinstance(unitless_val, sp.matrices.MatrixBase):
				unitless_val = tuple(
					parse_scalar(scalar)
					for scalar in unitless_val
				)
			else:
				unitless_val = parse_scalar(unitless_val)
			
			# Conservatively Set Differing Values
			if bl_modifier[interface_item.identifier] != unitless_val:
				bl_modifier[interface_item.identifier] = unitless_val
			
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
		bl_object = self.compute_input("preview_target")
		if bl_object is None: return
		
		# Remove Existing GeoNodes Modifier
		if GEONODES_MODIFIER_NAME in bl_object.modifiers:
			modifier_to_remove = bl_object.modifiers[GEONODES_MODIFIER_NAME]
			bl_object.modifiers.remove(modifier_to_remove)
		
		# Retrieve GeoNodes Tree
		geo_nodes = self.compute_input("geo_nodes")
		if geo_nodes is None: return
		
		# Add Non-Static Sockets from GeoNodes
		for bl_socket_name, bl_socket in geo_nodes.interface.items_tree.items():
			# For now, don't allow Geometry inputs.
			if bl_socket.socket_type == "NodeSocketGeometry": continue
			
			# Establish Dimensions of GeoNodes Input Sockets
			if (
				bl_socket.description.startswith("2D")
			):
				dimensions = 2
			elif (
				bl_socket.socket_type.startswith("NodeSocketVector")
				or bl_socket.socket_type.startswith("NodeSocketColor")
				or bl_socket.socket_type.startswith("NodeSocketRotation")
			):
				dimensions = 3
			else:
				dimensions = 1
			
			# Choose Socket via. Description Hint (if exists)
			if (
				":" in bl_socket.description
				and "(" in (desc_hint := bl_socket.description.split(":")[0])
				and ")" in desc_hint
			):
				for tag in contracts.BLNodeSocket_to_SocketType_by_desc[
					dimensions
				]:
					if desc_hint.startswith(tag):
						self.inputs.new(
							contracts.BLNodeSocket_to_SocketType_by_desc[
								dimensions
							][tag],
							bl_socket_name,
						)
						
						if len([
							(unit := _unit)
							for _unit in contracts.SocketType_to_units[
								contracts.SocketType[
									self.inputs[bl_socket_name].bl_idname.removesuffix("SocketType")
								]
							]["values"].values()
							if desc_hint[
								desc_hint.find("(")+1 : desc_hint.find(")")
							] == str(_unit)
						]) > 0:
							self.inputs[bl_socket_name].unit = unit
					
			elif bl_socket.socket_type in contracts.BLNodeSocket_to_SocketType[
				dimensions
			]:
				self.inputs.new(
					contracts.BLNodeSocket_to_SocketType[
						dimensions
					][bl_socket.socket_type],
					bl_socket_name,
				)
			
		
		# Create New GeoNodes Modifier
		if GEONODES_MODIFIER_NAME not in bl_object.modifiers:
			modifier = bl_object.modifiers.new(
				name=GEONODES_MODIFIER_NAME,
				type="NODES",
			)
			modifier.node_group = geo_nodes
		
		# Set Default Values
		for interface_item in geo_nodes.interface.items_tree.values():
			if (
				interface_item.name in self.inputs
				and hasattr(interface_item, "default_value") 
			):
				bl_socket = self.inputs[
					interface_item.name
				]
				if hasattr(bl_socket, "use_units"):
					bl_unit_system = self.compute_input("blender_unit_system")
					socket_type = contracts.SocketType[
						bl_socket.bl_idname.removesuffix("SocketType")
					]
					
					bl_socket.default_value = (
						parse_bl_to_sp(interface_item.default_value)
						* bl_unit_system[socket_type]
					)
				else:
					bl_socket.default_value = parse_bl_to_sp(interface_item.default_value)



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
