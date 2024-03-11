import typing as typ

import tidy3d as td
import numpy as np
import sympy as sp
import sympy.physics.units as spu

import bpy
from bpy_types import bpy_types
import bmesh

from .....utils import analyze_geonodes
from ... import bl_socket_map
from ... import contracts as ct
from ... import sockets
from .. import base
from ... import managed_objs

class GeoNodesStructureNode(base.MaxwellSimNode):
	node_type = ct.NodeType.GeoNodesStructure
	bl_label = "GeoNodes Structure"
	
	####################
	# - Sockets
	####################
	input_sockets = {
		"Unit System": sockets.PhysicalUnitSystemSocketDef(),
		"Medium": sockets.MaxwellMediumSocketDef(),
		"GeoNodes": sockets.BlenderGeoNodesSocketDef(),
	}
	output_sockets = {
		"Structure": sockets.MaxwellStructureSocketDef(),
	}
	
	managed_obj_defs = {
		"geometry": ct.schemas.ManagedObjDef(
			mk=lambda name: managed_objs.ManagedBLObject(name),
			name_prefix="geonodes_",
		)
	}
	
	####################
	# - Output Socket Computation
	####################
	@base.computes_output_socket(
		"Structure",
		input_sockets={"Medium"},
		managed_objs={"geometry"},
	)
	def compute_structure(
		self,
		input_sockets: dict[str, typ.Any],
		managed_objs: dict[str, ct.schemas.ManagedObj],
	) -> td.Structure:
		# Extract the Managed Blender Object
		mobj = managed_objs["geometry"]
		
		# Extract Geometry as Arrays
		geometry_as_arrays = mobj.mesh_as_arrays
		
		# Return TriMesh Structure
		return td.Structure(
			geometry=td.TriangleMesh.from_vertices_faces(
				geometry_as_arrays["verts"],
				geometry_as_arrays["faces"],
			),
			medium=input_sockets["Medium"],
		)
	
	####################
	# - Event Methods
	####################
	@base.on_value_changed(
		socket_name="GeoNodes",
		
		managed_objs={"geometry"},
		input_sockets={"GeoNodes"},
	)
	def on_value_changed__geonodes(
		self,
		managed_objs: dict[str, ct.schemas.ManagedObj],
		input_sockets: dict[str, typ.Any],
	) -> None:
		"""Called whenever the GeoNodes socket is changed.
		
		Refreshes the Loose Input Sockets, which map directly to the GeoNodes tree input sockets.
		"""
		if not (geo_nodes := input_sockets["GeoNodes"]):
			managed_objs["geometry"].free()
			self.loose_input_sockets = {}
			return
		
		# Analyze GeoNodes
		## Extract Valid Inputs (via GeoNodes Tree "Interface")
		geonodes_interface = analyze_geonodes.interface(
			geo_nodes, direc="INPUT"
		)
		
		# Set Loose Input Sockets
		## Retrieve the appropriate SocketDef for the Blender Interface Socket
		self.loose_input_sockets = {
			socket_name: bl_socket_map.socket_def_from_bl_interface_socket(
				bl_interface_socket
			)()  ## === <SocketType>SocketDef(), but with dynamic SocketDef
			for socket_name, bl_interface_socket in geonodes_interface.items()
		}
		
		## Set Loose `socket.value` from Interface `default_value`
		for socket_name in self.loose_input_sockets:
			socket = self.inputs[socket_name]
			bl_interface_socket = geonodes_interface[socket_name]
			
			socket.value = bl_socket_map.value_from_bl(bl_interface_socket)
		
		## Implicitly triggers the loose-input `on_value_changed` for each.
	
	@base.on_value_changed(
		any_loose_input_socket=True,
		
		managed_objs={"geometry"},
		input_sockets={"Unit System", "GeoNodes"},
	)
	def on_value_changed__loose_inputs(
		self,
		managed_objs: dict[str, ct.schemas.ManagedObj],
		input_sockets: dict[str, typ.Any],
		loose_input_sockets: dict[str, typ.Any],
	):
		"""Called whenever a Loose Input Socket is altered.
		
		Synchronizes the change to the actual GeoNodes modifier, so that the change is immediately visible.
		"""
		# Retrieve Data
		unit_system = input_sockets["Unit System"]
		mobj = managed_objs["geometry"]
		
		if not (geo_nodes := input_sockets["GeoNodes"]): return
		
		# Analyze GeoNodes Interface (input direction)
		## This retrieves NodeTreeSocketInterface elements
		geonodes_interface = analyze_geonodes.interface(
			geo_nodes, direc="INPUT"
		)
		
		## TODO: Check that Loose Sockets matches the Interface
		## - If the user deletes an interface socket, bad things will happen.
		## - We will try to set an identifier that doesn't exist!
		## - Instead, this should update the loose input sockets.
		
		## Push Values to the GeoNodes Modifier
		mobj.sync_geonodes_modifier(
			geonodes_node_group=geo_nodes,
			geonodes_identifier_to_value={
				bl_interface_socket.identifier: bl_socket_map.value_to_bl(
					bl_interface_socket,
					loose_input_sockets[socket_name],
					unit_system,
				)
				for socket_name, bl_interface_socket in (
					geonodes_interface.items()
				)
			}
		)
	
	####################
	# - Event Methods
	####################
	@base.on_show_preview(
		managed_objs={"geometry"},
	)
	def on_show_preview(
		self,
		managed_objs: dict[str, ct.schemas.ManagedObj],
	):
		"""Called whenever a Loose Input Socket is altered.
		
		Synchronizes the change to the actual GeoNodes modifier, so that the change is immediately visible.
		"""
		managed_objs["geometry"].show_preview("MESH")


####################
# - Blender Registration
####################
BL_REGISTER = [
	GeoNodesStructureNode,
]
BL_NODES = {
	ct.NodeType.GeoNodesStructure: (
		ct.NodeCategory.MAXWELLSIM_STRUCTURES
	)
}
