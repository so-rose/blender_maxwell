import bpy
import sympy as sp
import sympy.physics.units as spu
import scipy as sc

from .....utils import analyze_geonodes
from ... import contracts as ct
from ... import sockets
from .. import base
from ... import managed_objs

GEONODES_DOMAIN_BOX = "simdomain_box"

class SimDomainNode(base.MaxwellSimNode):
	node_type = ct.NodeType.SimDomain
	bl_label = "Sim Domain"
	
	input_sockets = {
		"Duration": sockets.PhysicalTimeSocketDef(
			default_value = 5 * spu.ps,
			default_unit = spu.ps,
		),
		"Center": sockets.PhysicalSize3DSocketDef(),
		"Size": sockets.PhysicalSize3DSocketDef(),
		"Grid": sockets.MaxwellSimGridSocketDef(),
		"Ambient Medium": sockets.MaxwellMediumSocketDef(),
	}
	output_sockets = {
		"Domain": sockets.MaxwellSimDomainSocketDef(),
	}
	
	managed_obj_defs = {
		"domain_box": ct.schemas.ManagedObjDef(
			mk=lambda name: managed_objs.ManagedBLObject(name),
			name_prefix="",
		)
	}
	
	####################
	# - Callbacks
	####################
	@base.computes_output_socket(
		"Domain",
		input_sockets={"Duration", "Center", "Size", "Grid", "Ambient Medium"},
	)
	def compute_sim_domain(self, input_sockets: dict) -> sp.Expr:
		if all([
			(_duration := input_sockets["Duration"]),
			(_center := input_sockets["Center"]),
			(_size := input_sockets["Size"]),
			(grid := input_sockets["Grid"]),
			(medium := input_sockets["Ambient Medium"]),
		]):
			duration = spu.convert_to(_duration, spu.second) / spu.second
			center = tuple(spu.convert_to(_center, spu.um) / spu.um)
			size = tuple(spu.convert_to(_size, spu.um) / spu.um)
			return dict(
				run_time=duration,
				center=center,
				size=size,
				grid_spec=grid,
				medium=medium,
			)
	
	####################
	# - Preview
	####################
	@base.on_value_changed(
		socket_name={"Center", "Size"},
		input_sockets={"Center", "Size"},
		managed_objs={"domain_box"},
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
		geo_nodes = bpy.data.node_groups[GEONODES_DOMAIN_BOX]
		geonodes_interface = analyze_geonodes.interface(
			geo_nodes, direc="INPUT"
		)
		
		# Sync Modifier Inputs
		managed_objs["domain_box"].sync_geonodes_modifier(
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
		managed_objs["domain_box"].bl_object("MESH").location = center
	
	@base.on_show_preview(
		managed_objs={"domain_box"},
	)
	def on_show_preview(
		self,
		managed_objs: dict[str, ct.schemas.ManagedObj],
	):
		managed_objs["domain_box"].show_preview("MESH")
		self.on_value_changed__center_size()

####################
# - Blender Registration
####################
BL_REGISTER = [
	SimDomainNode,
]
BL_NODES = {
	ct.NodeType.SimDomain: (
		ct.NodeCategory.MAXWELLSIM_SIMS
	)
}
