import bpy
import sympy.physics.units as spu
import tidy3d as td

from ......utils import analyze_geonodes
from .... import contracts as ct
from .... import managed_objs, sockets
from ... import base, events

GEONODES_STRUCTURE_SPHERE = 'structure_sphere'


class SphereStructureNode(base.MaxwellSimNode):
	node_type = ct.NodeType.SphereStructure
	bl_label = 'Sphere Structure'

	####################
	# - Sockets
	####################
	input_sockets = {
		'Center': sockets.PhysicalPoint3DSocketDef(),
		'Radius': sockets.PhysicalLengthSocketDef(
			default_value=150 * spu.nm,
		),
		'Medium': sockets.MaxwellMediumSocketDef(),
	}
	output_sockets = {
		'Structure': sockets.MaxwellStructureSocketDef(),
	}

	managed_obj_defs = {
		'structure_sphere': ct.schemas.ManagedObjDef(
			mk=lambda name: managed_objs.ManagedBLObject(name),
			name_prefix='',
		)
	}

	####################
	# - Output Socket Computation
	####################
	@events.computes_output_socket(
		'Structure',
		input_sockets={'Center', 'Radius', 'Medium'},
	)
	def compute_structure(self, input_sockets: dict) -> td.Box:
		medium = input_sockets['Medium']
		_center = input_sockets['Center']
		_radius = input_sockets['Radius']

		center = tuple(spu.convert_to(_center, spu.um) / spu.um)
		radius = spu.convert_to(_radius, spu.um) / spu.um

		return td.Structure(
			geometry=td.Sphere(
				radius=radius,
				center=center,
			),
			medium=medium,
		)

	####################
	# - Preview - Changes to Input Sockets
	####################
	@events.on_value_changed(
		socket_name={'Center', 'Radius'},
		input_sockets={'Center', 'Radius'},
		managed_objs={'structure_sphere'},
	)
	def on_value_changed__center_radius(
		self,
		input_sockets: dict,
		managed_objs: dict[str, ct.schemas.ManagedObj],
	):
		_center = input_sockets['Center']
		center = tuple([float(el) for el in spu.convert_to(_center, spu.um) / spu.um])

		_radius = input_sockets['Radius']
		radius = float(spu.convert_to(_radius, spu.um) / spu.um)
		## TODO: Preview unit system?? Presume um for now

		# Retrieve Hard-Coded GeoNodes and Analyze Input
		geo_nodes = bpy.data.node_groups[GEONODES_STRUCTURE_SPHERE]
		geonodes_interface = analyze_geonodes.interface(geo_nodes, direc='INPUT')

		# Sync Modifier Inputs
		managed_objs['structure_sphere'].sync_geonodes_modifier(
			geonodes_node_group=geo_nodes,
			geonodes_identifier_to_value={
				geonodes_interface['Radius'].identifier: radius,
				## TODO: Use 'bl_socket_map.value_to_bl`!
				## - This accounts for auto-conversion, unit systems, etc. .
				## - We could keep it in the node base class...
				## - ...But it needs aligning with Blender, too. Hmm.
			},
		)

		# Sync Object Position
		managed_objs['structure_sphere'].bl_object('MESH').location = center

	####################
	# - Preview - Show Preview
	####################
	@events.on_show_preview(
		managed_objs={'structure_sphere'},
	)
	def on_show_preview(
		self,
		managed_objs: dict[str, ct.schemas.ManagedObj],
	):
		managed_objs['structure_sphere'].show_preview('MESH')
		self.on_value_changed__center_radius()


####################
# - Blender Registration
####################
BL_REGISTER = [
	SphereStructureNode,
]
BL_NODES = {
	ct.NodeType.SphereStructure: (ct.NodeCategory.MAXWELLSIM_STRUCTURES_PRIMITIVES)
}
