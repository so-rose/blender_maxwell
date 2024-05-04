import typing as typ

import sympy as sp
import sympy.physics.units as spu
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
		'Center': sockets.ExprSocketDef(
			shape=(3,),
			physical_type=spux.PhysicalType.Length,
		),
		'Size': sockets.ExprSocketDef(
			shape=(3,),
			physical_type=spux.PhysicalType.Length,
			default_value=sp.Matrix([1, 1, 1]),
		),
		'Samples/Space': sockets.ExprSocketDef(
			shape=(3,),
			mathtype=spux.MathType.Integer,
			default_value=sp.Matrix([10, 10, 10]),
		),
		'Direction': sockets.BoolSocketDef(),
	}
	input_socket_sets: typ.ClassVar = {
		'Freq Domain': {
			'Freqs': sockets.ExprSocketDef(
				active_kind=ct.FlowKind.LazyArrayRange,
				physical_type=spux.PhysicalType.Freq,
				default_unit=spux.THz,
				default_min=374.7406,  ## 800nm
				default_max=1498.962,  ## 200nm
				default_steps=100,
			),
		},
		'Time Domain': {
			'Time Range': sockets.ExprSocketDef(
				active_kind=ct.FlowKind.LazyArrayRange,
				physical_type=spux.PhysicalType.Time,
				default_unit=spu.picosecond,
				default_min=0,
				default_max=10,
				default_steps=2,
			),
			'Samples/Time': sockets.ExprSocketDef(
				mathtype=spux.MathType.Integer,
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
			freqs=input_sockets['Freqs'].realize_array,
			normal_dir='+' if input_sockets['Direction'] else '-',
		)

	####################
	# - Preview - Changes to Input Sockets
	####################
	@events.on_value_changed(
		# Trigger
		prop_name='preview_active',
		# Loaded
		managed_objs={'mesh'},
		props={'preview_active'},
	)
	def on_preview_changed(self, managed_objs, props):
		"""Enables/disables previewing of the GeoNodes-driven mesh, regardless of whether a particular GeoNodes tree is chosen."""
		mesh = managed_objs['mesh']

		# Push Preview State to Managed Mesh
		if props['preview_active']:
			mesh.show_preview()
		else:
			mesh.hide_preview()

	@events.on_value_changed(
		# Trigger
		socket_name={'Center', 'Size', 'Direction'},
		run_on_init=True,
		# Loaded
		managed_objs={'mesh', 'modifier'},
		input_sockets={'Center', 'Size', 'Direction'},
		unit_systems={'BlenderUnits': ct.UNITS_BLENDER},
		scale_input_sockets={
			'Center': 'BlenderUnits',
		},
	)
	def on_inputs_changed(
		self,
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
					'Direction': input_sockets['Direction'],
				},
			},
		)


####################
# - Blender Registration
####################
BL_REGISTER = [
	PowerFluxMonitorNode,
]
BL_NODES = {ct.NodeType.PowerFluxMonitor: (ct.NodeCategory.MAXWELLSIM_MONITORS)}
