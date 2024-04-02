import bpy
import sympy as sp
import sympy.physics.units as spu
import tidy3d as td

from .....utils import analyze_geonodes
from .....utils import extra_sympy_units as spux
from ... import contracts as ct
from ... import managed_objs, sockets
from .. import base, events

GEONODES_MONITOR_BOX = 'monitor_flux_box'


class FieldPowerFluxMonitorNode(base.MaxwellSimNode):
	node_type = ct.NodeType.FieldPowerFluxMonitor
	bl_label = 'Field Power Flux Monitor'
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
		'Direction': sockets.BoolSocketDef(),
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
	# - Properties
	####################

	####################
	# - UI
	####################
	def draw_props(self, context, layout):
		pass

	def draw_info(self, context, col):
		pass

	####################
	# - Output Sockets
	####################
	@events.computes_output_socket(
		'Monitor',
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
		props={'active_socket_set', 'sim_node_name'},
	)
	def compute_monitor(self, input_sockets: dict, props: dict) -> td.FieldTimeMonitor:
		_center = input_sockets['Center']
		_size = input_sockets['Size']
		_samples_space = input_sockets['Samples/Space']

		center = tuple(spu.convert_to(_center, spu.um) / spu.um)
		size = tuple(spu.convert_to(_size, spu.um) / spu.um)
		samples_space = tuple(_samples_space)

		direction = '+' if input_sockets['Direction'] else '-'

		if props['active_socket_set'] == 'Freq Domain':
			freqs = input_sockets['Freqs']

			return td.FluxMonitor(
				center=center,
				size=size,
				name=props['sim_node_name'],
				interval_space=samples_space,
				freqs=[
					float(spu.convert_to(freq, spu.hertz) / spu.hertz) for freq in freqs
				],
				normal_dir=direction,
			)
		else:  ## Time Domain
			_rec_start = input_sockets['Rec Start']
			_rec_stop = input_sockets['Rec Stop']
			samples_time = input_sockets['Samples/Time']

			rec_start = spu.convert_to(_rec_start, spu.second) / spu.second
			rec_stop = spu.convert_to(_rec_stop, spu.second) / spu.second

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
	@events.on_value_changed(
		socket_name={'Center', 'Size'},
		input_sockets={'Center', 'Size', 'Direction'},
		managed_objs={'monitor_box'},
	)
	def on_value_changed__center_size(
		self,
		input_sockets: dict,
		managed_objs: dict[str, ct.schemas.ManagedObj],
	):
		_center = input_sockets['Center']
		center = tuple([float(el) for el in spu.convert_to(_center, spu.um) / spu.um])

		_size = input_sockets['Size']
		size = tuple([float(el) for el in spu.convert_to(_size, spu.um) / spu.um])
		## TODO: Preview unit system?? Presume um for now

		# Retrieve Hard-Coded GeoNodes and Analyze Input
		geo_nodes = bpy.data.node_groups[GEONODES_MONITOR_BOX]
		geonodes_interface = analyze_geonodes.interface(geo_nodes, direc='INPUT')

		# Sync Modifier Inputs
		managed_objs['monitor_box'].sync_geonodes_modifier(
			geonodes_node_group=geo_nodes,
			geonodes_identifier_to_value={
				geonodes_interface['Size'].identifier: size,
				geonodes_interface['Direction'].identifier: input_sockets['Direction'],
				## TODO: Use 'bl_socket_map.value_to_bl`!
				## - This accounts for auto-conversion, unit systems, etc. .
				## - We could keep it in the node base class...
				## - ...But it needs aligning with Blender, too. Hmm.
			},
		)

		# Sync Object Position
		managed_objs['monitor_box'].bl_object('MESH').location = center

	####################
	# - Preview - Show Preview
	####################
	@events.on_show_preview(
		managed_objs={'monitor_box'},
	)
	def on_show_preview(
		self,
		managed_objs: dict[str, ct.schemas.ManagedObj],
	):
		managed_objs['monitor_box'].show_preview('MESH')
		self.on_value_changed__center_size()


####################
# - Blender Registration
####################
BL_REGISTER = [
	FieldPowerFluxMonitorNode,
]
BL_NODES = {ct.NodeType.FieldPowerFluxMonitor: (ct.NodeCategory.MAXWELLSIM_MONITORS)}
