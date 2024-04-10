import typing as typ

import bpy
import sympy as sp
import sympy.physics.units as spu
import tidy3d as td

from .....assets.import_geonodes import GeoNodes, import_geonodes
from .....utils import extra_sympy_units as spux
from .....utils import logger
from ... import contracts as ct
from ... import managed_objs, sockets
from .. import base, events

log = logger.get(__name__)


class PowerFluxMonitorNode(base.MaxwellSimNode):
	node_type = ct.NodeType.PowerFluxMonitor
	bl_label = 'Power Flux Monitor'
	use_sim_node_name = True

	####################
	# - Sockets
	####################
	input_sockets: typ.ClassVar = {
		'Center': sockets.PhysicalPoint3DSocketDef(),
		'Size': sockets.PhysicalSize3DSocketDef(),
		'Samples/Space': sockets.Integer3DVectorSocketDef(
			default_value=sp.Matrix([10, 10, 10])
		),
		'Direction': sockets.BoolSocketDef(),
	}
	input_socket_sets: typ.ClassVar = {
		'Freq Domain': {
			'Freqs': sockets.PhysicalFreqSocketDef(
				is_array=True,
			),
		},
		'Time Domain': {
			'Rec Start': sockets.PhysicalTimeSocketDef(),
			'Rec Stop': sockets.PhysicalTimeSocketDef(default_value=200 * spux.fs),
			'Samples/Time': sockets.IntegerNumberSocketDef(
				default_value=100,
			),
		},
	}
	output_sockets: typ.ClassVar = {
		'Monitor': sockets.MaxwellMonitorSocketDef(),
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
	# - Event Methods: Computation
	####################
	@events.computes_output_socket(
		'Monitor',
		props={'active_socket_set', 'sim_node_name'},
		input_sockets={
			'Rec Start',
			'Rec Stop',
			'Center',
			'Size',
			'Samples/Space',
			'Samples/Time',
			'Freqs',
			'Direction',
		},
		input_socket_kinds={
			'Freqs': ct.LazyDataValueRange,
		},
		unit_systems={'Tidy3DUnits': ct.UNITS_TIDY3D},
		scale_input_sockets={
			'Center': 'Tidy3DUnits',
			'Size': 'Tidy3DUnits',
			'Freqs': 'Tidy3DUnits',
			'Samples/Space': 'Tidy3DUnits',
			'Rec Start': 'Tidy3DUnits',
			'Rec Stop': 'Tidy3DUnits',
			'Samples/Time': 'Tidy3DUnits',
		},
	)
	def compute_monitor(self, input_sockets: dict, props: dict) -> td.FieldTimeMonitor:
		direction = '+' if input_sockets['Direction'] else '-'

		if props['active_socket_set'] == 'Freq Domain':
			log.info(
				'Computing FluxMonitor (name="%s") with center="%s", size="%s"',
				props['sim_node_name'],
				input_sockets['Center'],
				input_sockets['Size'],
			)
			return td.FluxMonitor(
				center=input_sockets['Center'],
				size=input_sockets['Size'],
				name=props['sim_node_name'],
				interval_space=input_sockets['Samples/Space'],
				freqs=input_sockets['Freqs'].realize().values,
				normal_dir=direction,
			)

		return td.FluxTimeMonitor(
			center=input_sockets['Center'],
			size=input_sockets['Size'],
			name=props['sim_node_name'],
			start=input_sockets['Rec Start'],
			stop=input_sockets['Rec Stop'],
			interval=input_sockets['Samples/Time'],
			interval_space=input_sockets['Samples/Space'],
			normal_dir=direction,
		)

	####################
	# - Preview - Changes to Input Sockets
	####################
	@events.on_value_changed(
		socket_name={'Center', 'Size'},
		prop_name='preview_active',
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
		managed_objs: dict[str, ct.schemas.ManagedObj],
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
	PowerFluxMonitorNode,
]
BL_NODES = {ct.NodeType.PowerFluxMonitor: (ct.NodeCategory.MAXWELLSIM_MONITORS)}
