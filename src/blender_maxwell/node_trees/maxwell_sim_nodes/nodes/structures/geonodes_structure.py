import typing as typ

import tidy3d as td

from .....utils import analyze_geonodes, logger
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
		'Medium': sockets.MaxwellMediumSocketDef(),
		'Center': sockets.PhysicalPoint3DSocketDef(),
		'GeoNodes': sockets.BlenderGeoNodesSocketDef(),
	}
	output_sockets: typ.ClassVar = {
		'Structure': sockets.MaxwellStructureSocketDef(),
	}

	managed_obj_defs: typ.ClassVar = {
		'mesh': ct.schemas.ManagedObjDef(
			mk=lambda name: managed_objs.ManagedBLMesh(name),
		),
		'modifier': ct.schemas.ManagedObjDef(
			mk=lambda name: managed_objs.ManagedBLModifier(name),
		),
	}

	####################
	# - Event Methods
	####################
	@events.computes_output_socket(
		'Structure',
		input_sockets={'Medium'},
		managed_objs={'geometry'},
	)
	def compute_output(
		self,
		input_sockets: dict[str, typ.Any],
		managed_objs: dict[str, ct.schemas.ManagedObj],
	) -> td.Structure:
		# Simulate Input Value Change
		## This ensures that the mesh has been re-computed.
		self.on_input_changed()

		## TODO: mesh_as_arrays might not take the Center into account.
		## - Alternatively, Tidy3D might have a way to transform?
		mesh_as_arrays = managed_objs['mesh'].mesh_as_arrays
		return td.Structure(
			geometry=td.TriangleMesh.from_vertices_faces(
				mesh_as_arrays['verts'],
				mesh_as_arrays['faces'],
			),
			medium=input_sockets['Medium'],
		)

	####################
	# - Event Methods
	####################
	@events.on_value_changed(
		socket_name='GeoNodes',
		prop_name='preview_active',
		any_loose_input_socket=True,
		# Method Data
		managed_objs={'mesh', 'modifier'},
		input_sockets={'GeoNodes'},
		# Unit System Scaling
		unit_systems={'BlenderUnits': ct.UNITS_BLENDER},
	)
	def on_input_changed(
		self,
		props: dict,
		managed_objs: dict[str, ct.schemas.ManagedObj],
		input_sockets: dict,
		loose_input_sockets: dict,
		unit_systems: dict,
	) -> None:
		# No GeoNodes: Remove Modifier (if any)
		if (geonodes := input_sockets['GeoNodes']) is None:
			if (
				managed_objs['modifier'].name
				in managed_objs['mesh'].bl_object().modifiers
			):
				log.info(
					'Removing Modifier "%s" from BLObject "%s"',
					managed_objs['modifier'].name,
					managed_objs['mesh'].name,
				)
				managed_objs['mesh'].bl_object().modifiers.remove(
					managed_objs['modifier'].name
				)

				# Reset Loose Input Sockets
				self.loose_input_sockets = {}
			return

		# No Loose Input Sockets: Create from GeoNodes Interface
		## TODO: Other reasons to trigger re-filling loose_input_sockets.
		if not loose_input_sockets:
			# Retrieve the GeoNodes Interface
			geonodes_interface = analyze_geonodes.interface(
				input_sockets['GeoNodes'], direc='INPUT'
			)

			# Fill the Loose Input Sockets
			log.info(
				'Initializing GeoNodes Structure Node "%s" from GeoNodes Group "%s"',
				self.bl_label,
				str(geonodes),
			)
			self.loose_input_sockets = {
				socket_name: bl_socket_map.socket_def_from_bl_socket(iface_socket)()
				for socket_name, iface_socket in geonodes_interface.items()
			}

			# Set Loose Input Sockets to Interface (Default) Values
			## Changing socket.value invokes recursion of this function.
			## The else: below ensures that only one push occurs.
			## (well, one push per .value set, which simplifies to one push)
			log.debug(
				'Setting Loose Input Sockets of "%s" to GeoNodes Defaults',
				self.bl_label,
			)
			for socket_name in self.loose_input_sockets:
				socket = self.inputs[socket_name]
				socket.value = bl_socket_map.read_bl_socket_default_value(
					geonodes_interface[socket_name]
				)
		else:
			# Push Loose Input Values to GeoNodes Modifier
			managed_objs['modifier'].bl_modifier(
				managed_objs['mesh'].bl_object(location=input_sockets['Center']),
				'NODES',
				{
					'node_group': input_sockets['GeoNodes'],
					'unit_system': unit_systems['BlenderUnits'],
					'inputs': loose_input_sockets,
				},
			)
			# Push Preview State
			if props['preview_active']:
				managed_objs['mesh'].show_preview()


####################
# - Blender Registration
####################
BL_REGISTER = [
	GeoNodesStructureNode,
]
BL_NODES = {ct.NodeType.GeoNodesStructure: (ct.NodeCategory.MAXWELLSIM_STRUCTURES)}
