import typing as typ

import sympy as sp
import sympy.physics.units as spu

from blender_maxwell.assets.geonodes import GeoNodes, import_geonodes
from blender_maxwell.utils import extra_sympy_units as spux
from blender_maxwell.utils import logger

from ... import contracts as ct
from ... import managed_objs, sockets
from .. import base, events

log = logger.get(__name__)


class SimDomainNode(base.MaxwellSimNode):
	node_type = ct.NodeType.SimDomain
	bl_label = 'Sim Domain'
	use_sim_node_name = True

	input_sockets: typ.ClassVar = {
		'Duration': sockets.ExprSocketDef(
			physical_type=spux.PhysicalType.Time,
			default_unit=spu.picosecond,
			default_value=5,
			abs_min=0,
		),
		'Center': sockets.ExprSocketDef(
			shape=(3,),
			mathtype=spux.MathType.Real,
			physical_type=spux.PhysicalType.Length,
			default_unit=spu.micrometer,
			default_value=sp.Matrix([0, 0, 0]),
		),
		'Size': sockets.ExprSocketDef(
			shape=(3,),
			mathtype=spux.MathType.Real,
			physical_type=spux.PhysicalType.Length,
			default_unit=spu.micrometer,
			default_value=sp.Matrix([1, 1, 1]),
			abs_min=0.001,
		),
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
	# - Outputs
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
	def compute_domain(self, input_sockets, unit_systems) -> sp.Expr:
		return {
			'run_time': input_sockets['Duration'],
			'center': input_sockets['Center'],
			'size': input_sockets['Size'],
			'grid_spec': input_sockets['Grid'],
			'medium': input_sockets['Ambient Medium'],
		}

	####################
	# - Preview
	####################
	@events.on_value_changed(
		prop_name='preview_active',
		run_on_init=True,
		props={'preview_active'},
		managed_objs={'mesh'},
	)
	def on_preview_changed(self, props, managed_objs) -> None:
		mesh = managed_objs['mesh']

		# Push Preview State to Managed Mesh
		if props['preview_active']:
			mesh.show_preview()
		else:
			mesh.hide_preview()

	@events.on_value_changed(
		## Trigger
		socket_name={'Center', 'Size'},
		run_on_init=True,
		# Loaded
		input_sockets={'Center', 'Size'},
		managed_objs={'mesh', 'modifier'},
		unit_systems={'BlenderUnits': ct.UNITS_BLENDER},
		scale_input_sockets={
			'Center': 'BlenderUnits',
		},
	)
	def on_input_changed(
		self,
		managed_objs,
		input_sockets,
		unit_systems,
	):
		mesh = managed_objs['mesh']
		modifier = managed_objs['modifier']
		center = input_sockets['Center']
		size = input_sockets['Size']
		unit_system = unit_systems['BlenderUnits']

		# Push Loose Input Values to GeoNodes Modifier
		modifier.bl_modifier(
			mesh.bl_object(location=center),
			'NODES',
			{
				'node_group': import_geonodes(GeoNodes.SimulationSimDomain),
				'inputs': {'Size': size},
				'unit_system': unit_system,
			},
		)


####################
# - Blender Registration
####################
BL_REGISTER = [
	SimDomainNode,
]
BL_NODES = {ct.NodeType.SimDomain: (ct.NodeCategory.MAXWELLSIM_SIMS)}
