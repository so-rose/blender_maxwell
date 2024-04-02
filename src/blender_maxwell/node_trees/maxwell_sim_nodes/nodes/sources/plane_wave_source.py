import math

import bpy
import sympy as sp
import sympy.physics.units as spu
import tidy3d as td

from .....utils import analyze_geonodes
from ... import contracts as ct
from ... import managed_objs, sockets
from .. import base, events

GEONODES_PLANE_WAVE = 'source_plane_wave'


def convert_vector_to_spherical(
	v: sp.MatrixBase,
) -> tuple[str, str, sp.Expr, sp.Expr]:
	"""Converts a vector (maybe normalized) to spherical coordinates from an arbitrary choice of injection axis.

	Injection axis is chosen to minimize `theta`
	"""
	x, y, z = v

	injection_axis = max(
		('x', abs(x)), ('y', abs(y)), ('z', abs(z)), key=lambda item: item[1]
	)[0]
	## Select injection axis that minimizes 'theta'

	if injection_axis == 'x':
		direction = '+' if x >= 0 else '-'
		theta = sp.acos(x / sp.sqrt(x**2 + y**2 + z**2))
		phi = sp.atan2(z, y)
	elif injection_axis == 'y':
		direction = '+' if y >= 0 else '-'
		theta = sp.acos(y / sp.sqrt(x**2 + y**2 + z**2))
		phi = sp.atan2(x, z)
	else:
		direction = '+' if z >= 0 else '-'
		theta = sp.acos(z / sp.sqrt(x**2 + y**2 + z**2))
		phi = sp.atan2(y, x)

	return injection_axis, direction, theta, phi


class PlaneWaveSourceNode(base.MaxwellSimNode):
	node_type = ct.NodeType.PlaneWaveSource
	bl_label = 'Plane Wave Source'

	####################
	# - Sockets
	####################
	input_sockets = {
		'Temporal Shape': sockets.MaxwellTemporalShapeSocketDef(),
		'Center': sockets.PhysicalPoint3DSocketDef(),
		'Direction': sockets.Real3DVectorSocketDef(default_value=sp.Matrix([0, 0, -1])),
		'Pol Angle': sockets.PhysicalAngleSocketDef(),
	}
	output_sockets = {
		'Source': sockets.MaxwellSourceSocketDef(),
	}

	managed_obj_defs = {
		'plane_wave_source': ct.schemas.ManagedObjDef(
			mk=lambda name: managed_objs.ManagedBLObject(name),
			name_prefix='',
		)
	}

	####################
	# - Output Socket Computation
	####################
	@events.computes_output_socket(
		'Source',
		input_sockets={'Temporal Shape', 'Center', 'Direction', 'Pol Angle'},
	)
	def compute_source(self, input_sockets: dict):
		temporal_shape = input_sockets['Temporal Shape']
		_center = input_sockets['Center']
		direction = input_sockets['Direction']
		pol_angle = input_sockets['Pol Angle']

		injection_axis, dir_sgn, theta, phi = convert_vector_to_spherical(direction)

		size = {
			'x': (0, math.inf, math.inf),
			'y': (math.inf, 0, math.inf),
			'z': (math.inf, math.inf, 0),
		}[injection_axis]
		center = tuple(spu.convert_to(_center, spu.um) / spu.um)

		# Display the results
		return td.PlaneWave(
			center=center,
			source_time=temporal_shape,
			size=size,
			direction=dir_sgn,
			angle_theta=theta,
			angle_phi=phi,
			pol_angle=pol_angle,
		)

	####################
	# - Preview
	####################
	@events.on_value_changed(
		socket_name={'Center', 'Direction'},
		input_sockets={'Center', 'Direction'},
		managed_objs={'plane_wave_source'},
	)
	def on_value_changed__center_direction(
		self,
		input_sockets: dict,
		managed_objs: dict[str, ct.schemas.ManagedObj],
	):
		_center = input_sockets['Center']
		center = tuple([float(el) for el in spu.convert_to(_center, spu.um) / spu.um])

		_direction = input_sockets['Direction']
		direction = tuple([float(el) for el in _direction])
		## TODO: Preview unit system?? Presume um for now

		# Retrieve Hard-Coded GeoNodes and Analyze Input
		geo_nodes = bpy.data.node_groups[GEONODES_PLANE_WAVE]
		geonodes_interface = analyze_geonodes.interface(geo_nodes, direc='INPUT')

		# Sync Modifier Inputs
		managed_objs['plane_wave_source'].sync_geonodes_modifier(
			geonodes_node_group=geo_nodes,
			geonodes_identifier_to_value={
				geonodes_interface['Direction'].identifier: direction,
				## TODO: Use 'bl_socket_map.value_to_bl`!
				## - This accounts for auto-conversion, unit systems, etc. .
				## - We could keep it in the node base class...
				## - ...But it needs aligning with Blender, too. Hmm.
			},
		)

		# Sync Object Position
		managed_objs['plane_wave_source'].bl_object('MESH').location = center

	@events.on_show_preview(
		managed_objs={'plane_wave_source'},
	)
	def on_show_preview(
		self,
		managed_objs: dict[str, ct.schemas.ManagedObj],
	):
		managed_objs['plane_wave_source'].show_preview('MESH')
		self.on_value_changed__center_direction()


####################
# - Blender Registration
####################
BL_REGISTER = [
	PlaneWaveSourceNode,
]
BL_NODES = {ct.NodeType.PlaneWaveSource: (ct.NodeCategory.MAXWELLSIM_SOURCES)}
