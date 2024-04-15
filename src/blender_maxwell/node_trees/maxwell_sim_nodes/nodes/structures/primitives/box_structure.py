import typing as typ

import sympy as sp
import sympy.physics.units as spu
import tidy3d as td

from ......assets.import_geonodes import GeoNodes, import_geonodes
from .... import contracts as ct
from .... import managed_objs, sockets
from ... import base, events


class BoxStructureNode(base.MaxwellSimNode):
	node_type = ct.NodeType.BoxStructure
	bl_label = 'Box Structure'
	use_sim_node_name = True

	####################
	# - Sockets
	####################
	input_sockets: typ.ClassVar = {
		'Medium': sockets.MaxwellMediumSocketDef(),
		'Center': sockets.PhysicalPoint3DSocketDef(),
		'Size': sockets.PhysicalSize3DSocketDef(
			default_value=sp.Matrix([500, 500, 500]) * spu.nm
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
	# - Event Methods
	####################
	@events.computes_output_socket(
		'Structure',
		input_sockets={'Medium', 'Center', 'Size'},
		unit_systems={'Tidy3DUnits': ct.UNITS_TIDY3D},
		scale_input_sockets={
			'Center': 'Tidy3DUnits',
			'Size': 'Tidy3DUnits',
		},
	)
	def compute_structure(self, input_sockets: dict, unit_systems: dict) -> td.Box:
		return td.Structure(
			geometry=td.Box(
				center=input_sockets['Center'],
				size=input_sockets['Size'],
			),
			medium=input_sockets['Medium'],
		)

	@events.on_value_changed(
		socket_name={'Center', 'Size'},
		prop_name='preview_active',
		run_on_init=True,
		props={'preview_active'},
		input_sockets={'Center', 'Size'},
		managed_objs={'mesh', 'modifier'},
		unit_systems={'BlenderUnits': ct.UNITS_BLENDER},
		scale_input_sockets={
			'Center': 'BlenderUnits',
		},
	)
	def on_inputs_changed(
		self,
		props: dict,
		managed_objs: dict,
		input_sockets: dict,
		unit_systems: dict,
	):
		# Push Input Values to GeoNodes Modifier
		managed_objs['modifier'].bl_modifier(
			managed_objs['mesh'].bl_object(location=input_sockets['Center']),
			'NODES',
			{
				'node_group': import_geonodes(GeoNodes.PrimitiveBox, 'link'),
				'unit_system': unit_systems['BlenderUnits'],
				'inputs': {
					'Size': input_sockets['Size'],
				},
			},
		)
		# Push Preview State
		if props['preview_active']:
			managed_objs['mesh'].show_preview()


####################
# - Blender Registration
####################
BL_REGISTER = [
	BoxStructureNode,
]
BL_NODES = {
	ct.NodeType.BoxStructure: (ct.NodeCategory.MAXWELLSIM_STRUCTURES_PRIMITIVES)
}
