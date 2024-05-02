import typing as typ

import sympy as sp
import sympy.physics.units as spu
import tidy3d as td

from blender_maxwell.assets.geonodes import GeoNodes, import_geonodes
from blender_maxwell.utils import extra_sympy_units as spux
from blender_maxwell.utils import logger

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
		'GeoNodes': sockets.BlenderGeoNodesSocketDef(),
		'Medium': sockets.MaxwellMediumSocketDef(),
		'Center': sockets.ExprSocketDef(
			shape=(3,),
			mathtype=spux.MathType.Real,
			physical_type=spux.PhysicalType.Length,
			default_unit=spu.micrometer,
			default_value=sp.Matrix([0, 0, 0]),
		),
	}
	output_sockets: typ.ClassVar = {
		'Structure': sockets.MaxwellStructureSocketDef(),
	}

	managed_obj_types: typ.ClassVar = {
		'mesh': managed_objs.ManagedBLMesh,
		'modifier': managed_objs.ManagedBLModifier,
	}

	####################
	# - Outputs
	####################
	@events.computes_output_socket(
		'Structure',
		input_sockets={'Medium'},
		managed_objs={'mesh'},
	)
	def compute_structure(
		self,
		input_sockets,
		managed_objs,
	) -> td.Structure:
		"""Computes a triangle-mesh based Tidy3D structure, by manually copying mesh data from Blender to a `td.TriangleMesh`."""
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
	# - Events: Preview Active Changed
	####################
	@events.on_value_changed(
		prop_name='preview_active',
		props={'preview_active'},
		managed_objs={'mesh'},
	)
	def on_preview_changed(self, props) -> None:
		"""Enables/disables previewing of the GeoNodes-driven mesh, regardless of whether a particular GeoNodes tree is chosen."""
		mesh = managed_objs['mesh']

		# Push Preview State to Managed Mesh
		if props['preview_active']:
			mesh.show_preview()
		else:
			mesh.hide_preview()

	####################
	# - Events: GN Input Changed
	####################
	@events.on_value_changed(
		socket_name={'Center'},
		any_loose_input_socket=True,
		# Pass Data
		managed_objs={'mesh', 'modifier'},
		input_sockets={'Center', 'GeoNodes'},
		all_loose_input_sockets=True,
		unit_systems={'BlenderUnits': ct.UNITS_BLENDER},
		scale_input_sockets={'Center': 'BlenderUnits'},
	)
	def on_input_socket_changed(
		self, input_sockets, loose_input_sockets, unit_systems
	) -> None:
		"""Pushes any change in GeoNodes-bound input sockets to the GeoNodes modifier.

		Also pushes the `Center:Value` socket to govern the object's center in 3D space.
		"""
		geonodes = input_sockets['GeoNodes']
		has_geonodes = not ct.FlowSignal.check(geonodes)

		if has_geonodes:
			mesh = managed_objs['mesh']
			modifier = managed_objs['modifier']
			center = input_sockets['Center']
			unit_system = unit_systems['BlenderUnits']

			# Push Loose Input Values to GeoNodes Modifier
			modifier.bl_modifier(
				mesh.bl_object(location=center),
				'NODES',
				{
					'node_group': geonodes,
					'inputs': loose_input_sockets,
					'unit_system': unit_system,
				},
			)

	####################
	# - Events: GN Tree Changed
	####################
	@events.on_value_changed(
		socket_name={'GeoNodes'},
		# Pass Data
		managed_objs={'mesh', 'modifier'},
		input_sockets={'GeoNodes', 'Center'},
	)
	def on_input_changed(
		self,
		managed_objs,
		input_sockets,
	) -> None:
		"""Declares new loose input sockets in response to a new GeoNodes tree (if any)."""
		geonodes = input_sockets['GeoNodes']
		has_geonodes = not ct.FlowSignal.check(geonodes)

		if has_geonodes:
			mesh = managed_objs['mesh']
			modifier = managed_objs['modifier']

			# Fill the Loose Input Sockets
			## -> The SocketDefs contain the default values from the interface.
			log.info(
				'Initializing GeoNodes Structure Node "%s" from GeoNodes Group "%s"',
				self.bl_label,
				str(geonodes),
			)
			self.loose_input_sockets = bl_socket_map.sockets_from_geonodes(geonodes)

			## -> The loose socket creation triggers 'on_input_socket_changed'

		elif self.loose_input_sockets:
			self.loose_input_sockets = {}

			if modifier.name in mesh.bl_object().modifiers.keys().copy():
				modifier.free_from_bl_object(mesh.bl_object())


####################
# - Blender Registration
####################
BL_REGISTER = [
	GeoNodesStructureNode,
]
BL_NODES = {ct.NodeType.GeoNodesStructure: (ct.NodeCategory.MAXWELLSIM_STRUCTURES)}
