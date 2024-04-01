import bpy
import sympy as sp
import sympy.physics.units as spu
import tidy3d as td

from .....utils import analyze_geonodes, logger
from .....utils import extra_sympy_units as spux
from ... import contracts as ct
from ... import managed_objs, sockets
from .. import base

log = logger.get(__name__)

GEONODES_MONITOR_BOX = 'monitor_box'


class EHFieldMonitorNode(base.MaxwellSimNode):
	"""Node providing for the monitoring of electromagnetic fields within a given planar region or volume."""

	node_type = ct.NodeType.EHFieldMonitor
	bl_label = 'E/H Field Monitor'
	use_sim_node_name = True

	####################
	# - Sockets
	####################
	input_sockets = {
		'Center': sockets.PhysicalPoint3DSocketDef(),
		'Size': sockets.PhysicalSize3DSocketDef(),
		'Samples/Space': sockets.Integer3DVectorSocketDef(
			default_value=sp.Matrix([10, 10, 10])
		),
	}
	input_socket_sets = {
		'Freq Domain': {
			'Freqs': sockets.PhysicalFreqSocketDef(
				is_list=True,
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
	output_sockets = {
		'Monitor': sockets.MaxwellMonitorSocketDef(),
	}

	managed_obj_defs = {
		'monitor_box': ct.schemas.ManagedObjDef(
			mk=lambda name: managed_objs.ManagedBLObject(name),
			name_prefix='',
		)
	}

	####################
	# - Output Sockets
	####################
	@base.computes_output_socket(
		'Monitor',
		input_sockets={
			'Rec Start',
			'Rec Stop',
			'Center',
			'Size',
			'Samples/Space',
			'Samples/Time',
			'Freqs',
		},
		props={'active_socket_set', 'sim_node_name'},
	)
	def compute_monitor(
		self, input_sockets: dict, props: dict
	) -> td.FieldMonitor | td.FieldTimeMonitor:
		"""Computes the value of the 'Monitor' output socket, which the user can select as being either a `td.FieldMonitor` or `td.FieldTimeMonitor`."""
		_center = input_sockets['Center']
		_size = input_sockets['Size']
		_samples_space = input_sockets['Samples/Space']

		center = tuple(spu.convert_to(_center, spu.um) / spu.um)
		size = tuple(spu.convert_to(_size, spu.um) / spu.um)
		samples_space = tuple(_samples_space)

		if props['active_socket_set'] == 'Freq Domain':
			freqs = input_sockets['Freqs']

			log.info(
				'Computing FieldMonitor (name=%s) with center=%s, size=%s',
				props['sim_node_name'],
				center,
				size,
			)
			return td.FieldMonitor(
				center=center,
				size=size,
				name=props['sim_node_name'],
				interval_space=samples_space,
				freqs=[
					float(spu.convert_to(freq, spu.hertz) / spu.hertz) for freq in freqs
				],
			)
		## Time Domain
		_rec_start = input_sockets['Rec Start']
		_rec_stop = input_sockets['Rec Stop']
		samples_time = input_sockets['Samples/Time']

		rec_start = spu.convert_to(_rec_start, spu.second) / spu.second
		rec_stop = spu.convert_to(_rec_stop, spu.second) / spu.second

		log.info(
			'Computing FieldTimeMonitor (name=%s) with center=%s, size=%s',
			props['sim_node_name'],
			center,
			size,
		)
		return td.FieldTimeMonitor(
			center=center,
			size=size,
			name=props['sim_node_name'],
			start=rec_start,
			stop=rec_stop,
			interval=samples_time,
			interval_space=samples_space,
		)

	####################
	# - Preview - Changes to Input Sockets
	####################
	@base.on_value_changed(
		socket_name={'Center', 'Size'},
		input_sockets={'Center', 'Size'},
		managed_objs={'monitor_box'},
	)
	def on_value_changed__center_size(
		self,
		input_sockets: dict,
		managed_objs: dict[str, ct.schemas.ManagedObj],
	):
		"""Alters the managed 3D preview objects whenever the center or size input sockets are changed."""
		_center = input_sockets['Center']
		center = tuple([float(el) for el in spu.convert_to(_center, spu.um) / spu.um])

		_size = input_sockets['Size']
		size = tuple([float(el) for el in spu.convert_to(_size, spu.um) / spu.um])

		# Retrieve Hard-Coded GeoNodes and Analyze Input
		geo_nodes = bpy.data.node_groups[GEONODES_MONITOR_BOX]
		geonodes_interface = analyze_geonodes.interface(geo_nodes, direc='INPUT')

		# Sync Modifier Inputs
		managed_objs['monitor_box'].sync_geonodes_modifier(
			geonodes_node_group=geo_nodes,
			geonodes_identifier_to_value={
				geonodes_interface['Size'].identifier: size,
			},
		)

		# Sync Object Position
		managed_objs['monitor_box'].bl_object('MESH').location = center

	####################
	# - Preview - Show Preview
	####################
	@base.on_show_preview(
		managed_objs={'monitor_box'},
	)
	def on_show_preview(
		self,
		managed_objs: dict[str, ct.schemas.ManagedObj],
	):
		"""Requests that the managed object be previewed in response to a user request to show the preview."""
		managed_objs['monitor_box'].show_preview('MESH')
		self.on_value_changed__center_size()


####################
# - Blender Registration
####################
BL_REGISTER = [
	EHFieldMonitorNode,
]
BL_NODES = {ct.NodeType.EHFieldMonitor: (ct.NodeCategory.MAXWELLSIM_MONITORS)}
