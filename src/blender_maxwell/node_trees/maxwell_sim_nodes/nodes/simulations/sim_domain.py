import typing as typ

import sympy as sp
import sympy.physics.units as spu

from .....assets.import_geonodes import GeoNodes, import_geonodes
from ... import contracts as ct
from ... import managed_objs, sockets
from .. import base, events


class SimDomainNode(base.MaxwellSimNode):
	node_type = ct.NodeType.SimDomain
	bl_label = 'Sim Domain'
	use_sim_node_name = True

	input_sockets: typ.ClassVar = {
		'Duration': sockets.PhysicalTimeSocketDef(
			default_value=5 * spu.ps,
			default_unit=spu.ps,
		),
		'Center': sockets.PhysicalPoint3DSocketDef(),
		'Size': sockets.PhysicalSize3DSocketDef(),
		'Grid': sockets.MaxwellSimGridSocketDef(),
		'Ambient Medium': sockets.MaxwellMediumSocketDef(),
	}
	output_sockets: typ.ClassVar = {
		'Domain': sockets.MaxwellSimDomainSocketDef(),
	}

	managed_obj_types: typ.ClassVar = {
		'mesh': managed_objs.ManagedBLMesh,
		'modifier': managed_objs.ManagedBLModifier,
	}

	####################
	# - Event Methods
	####################
	@events.computes_output_socket(
		'Domain',
		input_sockets={'Duration', 'Center', 'Size', 'Grid', 'Ambient Medium'},
		unit_systems={'Tidy3DUnits': ct.UNITS_TIDY3D},
		scale_input_sockets={
			'Duration': 'Tidy3DUnits',
			'Center': 'Tidy3DUnits',
			'Size': 'Tidy3DUnits',
		},
	)
	def compute_domain(self, input_sockets: dict, unit_systems) -> sp.Expr:
		return {
			'run_time': input_sockets['Duration'],
			'center': input_sockets['Center'],
			'size': input_sockets['Size'],
			'grid_spec': input_sockets['Grid'],
			'medium': input_sockets['Ambient Medium'],
		}

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
	def on_input_changed(
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
	SimDomainNode,
]
BL_NODES = {ct.NodeType.SimDomain: (ct.NodeCategory.MAXWELLSIM_SIMS)}
