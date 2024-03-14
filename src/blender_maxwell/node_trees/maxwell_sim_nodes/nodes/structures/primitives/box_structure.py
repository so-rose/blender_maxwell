import tidy3d as td
import sympy as sp
import sympy.physics.units as spu

import bpy

from ......utils import analyze_geonodes
from .... import contracts as ct
from .... import sockets
from .... import managed_objs
from ... import base

GEONODES_STRUCTURE_BOX = "structure_box"

class BoxStructureNode(base.MaxwellSimNode):
	node_type = ct.NodeType.BoxStructure
	bl_label = "Box Structure"
	
	####################
	# - Sockets
	####################
	input_sockets = {
		"Medium": sockets.MaxwellMediumSocketDef(),
		"Center": sockets.PhysicalPoint3DSocketDef(),
		"Size": sockets.PhysicalSize3DSocketDef(
			default_value=sp.Matrix([500, 500, 500]) * spu.nm
		),
	}
	output_sockets = {
		"Structure": sockets.MaxwellStructureSocketDef(),
	}
	
	managed_obj_defs = {
		"structure_box": ct.schemas.ManagedObjDef(
			mk=lambda name: managed_objs.ManagedBLObject(name),
			name_prefix="",
		)
	}
	
	####################
	# - Output Socket Computation
	####################
	@base.computes_output_socket(
		"Structure",
		input_sockets={"Medium", "Center", "Size"},
	)
	def compute_simulation(self, input_sockets: dict) -> td.Box:
		medium = input_sockets["Medium"]
		_center = input_sockets["Center"]
		_size = input_sockets["Size"]
		
		center = tuple(spu.convert_to(_center, spu.um) / spu.um)
		size = tuple(spu.convert_to(_size, spu.um) / spu.um)
		
		return td.Structure(
			geometry=td.Box(
				center=center,
				size=size,
			),
			medium=medium,
		)
	
	####################
	# - Preview - Changes to Input Sockets
	####################
	@base.on_value_changed(
		socket_name={"Center", "Size"},
		input_sockets={"Center", "Size"},
		managed_objs={"structure_box"},
	)
	def on_value_changed__center_size(
		self,
		input_sockets: dict,
		managed_objs: dict[str, ct.schemas.ManagedObj],
	):
		_center = input_sockets["Center"]
		center = tuple([
			float(el)
			for el in spu.convert_to(_center, spu.um) / spu.um
		])
		
		_size = input_sockets["Size"]
		size = tuple([
			float(el)
			for el in spu.convert_to(_size, spu.um) / spu.um
		])
		## TODO: Preview unit system?? Presume um for now
		
		# Retrieve Hard-Coded GeoNodes and Analyze Input
		geo_nodes = bpy.data.node_groups[GEONODES_STRUCTURE_BOX]
		geonodes_interface = analyze_geonodes.interface(
			geo_nodes, direc="INPUT"
		)
		
		# Sync Modifier Inputs
		managed_objs["structure_box"].sync_geonodes_modifier(
			geonodes_node_group=geo_nodes,
			geonodes_identifier_to_value={
				geonodes_interface["Size"].identifier: size,
				## TODO: Use 'bl_socket_map.value_to_bl`!
				## - This accounts for auto-conversion, unit systems, etc. .
				## - We could keep it in the node base class...
				## - ...But it needs aligning with Blender, too. Hmm.
			}
		)
		
		# Sync Object Position
		managed_objs["structure_box"].bl_object("MESH").location = center
	
	####################
	# - Preview - Show Preview
	####################
	@base.on_show_preview(
		managed_objs={"structure_box"},
	)
	def on_show_preview(
		self,
		managed_objs: dict[str, ct.schemas.ManagedObj],
	):
		managed_objs["structure_box"].show_preview("MESH")
		self.on_value_changed__center_size()



####################
# - Blender Registration
####################
BL_REGISTER = [
	BoxStructureNode,
]
BL_NODES = {
	ct.NodeType.BoxStructure: (
		ct.NodeCategory.MAXWELLSIM_STRUCTURES_PRIMITIVES
	)
}
