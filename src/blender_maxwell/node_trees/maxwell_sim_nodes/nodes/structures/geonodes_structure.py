import typing as typ

import tidy3d as td

from blender_maxwell.utils import analyze_geonodes, logger

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

	managed_obj_types: typ.ClassVar = {
		'mesh': managed_objs.ManagedBLMesh,
		'modifier': managed_objs.ManagedBLModifier,
	}

	####################
	# - Event Methods
	####################
	@events.computes_output_socket(
		'Structure',
		input_sockets={'Medium'},
		managed_objs={'mesh'},
	)
	def compute_structure(
		self,
		input_sockets: dict,
		managed_objs: dict,
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
		socket_name={'GeoNodes', 'Center'},
		prop_name='preview_active',
		any_loose_input_socket=True,
		run_on_init=True,
		# Pass Data
		props={'preview_active'},
		managed_objs={'mesh', 'modifier'},
		input_sockets={'Center', 'GeoNodes'},
		all_loose_input_sockets=True,
		unit_systems={'BlenderUnits': ct.UNITS_BLENDER},
		scale_input_sockets={'Center': 'BlenderUnits'},
	)
	def on_input_changed(
		self,
		props: dict,
		managed_objs: dict,
		input_sockets: dict,
		loose_input_sockets: dict,
		unit_systems: dict,
	) -> None:
		# No GeoNodes: Remove Modifier (if any)
		if (geonodes := input_sockets['GeoNodes']) is None:
			if (
				managed_objs['modifier'].name
				in managed_objs['mesh'].bl_object().modifiers.keys().copy()
			):
				managed_objs['modifier'].free_from_bl_object(
					managed_objs['mesh'].bl_object()
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
			log.info(
				'Setting Loose Input Sockets of "%s" to GeoNodes Defaults',
				self.bl_label,
			)
			for socket_name in self.loose_input_sockets:
				socket = self.inputs[socket_name]
				socket.value = bl_socket_map.read_bl_socket_default_value(
					geonodes_interface[socket_name],
					unit_systems['BlenderUnits'],
					allow_unit_not_in_unit_system=True,
				)
			log.info(
				'Set Loose Input Sockets of "%s" to: %s',
				self.bl_label,
				str(self.loose_input_sockets),
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
