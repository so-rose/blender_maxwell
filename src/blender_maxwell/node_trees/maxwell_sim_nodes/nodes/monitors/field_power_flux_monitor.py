import typing as typ

import sympy as sp
import tidy3d as td

from blender_maxwell.assets.geonodes import GeoNodes, import_geonodes
from blender_maxwell.utils import extra_sympy_units as spux
from blender_maxwell.utils import logger

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
	output_socket_sets: typ.ClassVar = {
		'Freq Domain': {'Freq Monitor': sockets.MaxwellMonitorSocketDef()},
		'Time Domain': {'Time Monitor': sockets.MaxwellMonitorSocketDef()},
	}

	managed_obj_types: typ.ClassVar = {
		'mesh': managed_objs.ManagedBLMesh,
		'modifier': managed_objs.ManagedBLModifier,
	}

	####################
	# - Event Methods: Computation
	####################
	@events.computes_output_socket(
		'Freq Monitor',
		props={'sim_node_name'},
		input_sockets={
			'Center',
			'Size',
			'Samples/Space',
			'Freqs',
			'Direction',
		},
		input_socket_kinds={
			'Freqs': ct.FlowKind.LazyArrayRange,
		},
		unit_systems={'Tidy3DUnits': ct.UNITS_TIDY3D},
		scale_input_sockets={
			'Center': 'Tidy3DUnits',
			'Size': 'Tidy3DUnits',
			'Freqs': 'Tidy3DUnits',
		},
	)
	def compute_freq_monitor(
		self,
		input_sockets: dict,
		props: dict,
		unit_systems: dict,
	) -> td.FieldMonitor:
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
			interval_space=(1, 1, 1),
			freqs=input_sockets['Freqs'].realize().values,
			normal_dir='+' if input_sockets['Direction'] else '-',
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
		managed_objs: dict,
		input_sockets: dict,
		unit_systems: dict,
	):
		# Push Input Values to GeoNodes Modifier
		managed_objs['modifier'].bl_modifier(
			managed_objs['mesh'].bl_object(location=input_sockets['Center']),
			'NODES',
			{
				'node_group': import_geonodes(GeoNodes.MonitorPowerFlux),
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
