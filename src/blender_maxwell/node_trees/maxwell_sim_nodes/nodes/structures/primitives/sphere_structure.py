import typing as typ

import sympy.physics.units as spu
import tidy3d as td

from blender_maxwell.assets.geonodes import GeoNodes, import_geonodes

from .... import contracts as ct
from .... import managed_objs, sockets
from ... import base, events


class SphereStructureNode(base.MaxwellSimNode):
	node_type = ct.NodeType.SphereStructure
	bl_label = 'Sphere Structure'
	use_sim_node_name = True

	####################
	# - Sockets
	####################
	input_sockets: typ.ClassVar = {
		'Medium': sockets.MaxwellMediumSocketDef(),
		'Center': sockets.PhysicalPoint3DSocketDef(),
		'Radius': sockets.PhysicalLengthSocketDef(
			default_value=150 * spu.nm,
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
	# - Output Socket Computation
	####################
	@events.computes_output_socket(
		'Structure',
		input_sockets={'Center', 'Radius', 'Medium'},
		unit_systems={'Tidy3DUnits': ct.UNITS_TIDY3D},
		scale_input_sockets={
			'Center': 'Tidy3DUnits',
			'Radius': 'Tidy3DUnits',
		},
	)
	def compute_structure(self, input_sockets: dict) -> td.Box:
		return td.Structure(
			geometry=td.Sphere(
				radius=input_sockets['Radius'],
				center=input_sockets['Center'],
			),
			medium=input_sockets['Medium'],
		)

	####################
	# - Preview - Changes to Input Sockets
	####################
	@events.on_value_changed(
		socket_name={'Center', 'Radius'},
		prop_name='preview_active',
		run_on_init=True,
		props={'preview_active'},
		input_sockets={'Center', 'Radius'},
		managed_objs={'mesh', 'modifier'},
		unit_systems={'BlenderUnits': ct.UNITS_BLENDER},
		scale_input_sockets={
			'Center': 'Tidy3DUnits',
			'Radius': 'Tidy3DUnits',
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
				'node_group': import_geonodes(GeoNodes.StructurePrimitiveSphere),
				'unit_system': unit_systems['BlenderUnits'],
				'inputs': {
					'Radius': input_sockets['Radius'],
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
	SphereStructureNode,
]
BL_NODES = {
	ct.NodeType.SphereStructure: (ct.NodeCategory.MAXWELLSIM_STRUCTURES_PRIMITIVES)
}
