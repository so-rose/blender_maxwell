import math

import tidy3d as td
import sympy as sp
import sympy.physics.units as spu

import bpy

from ... import contracts as ct
from ... import sockets
from .. import base

class PlaneWaveSourceNode(base.MaxwellSimNode):
	node_type = ct.NodeType.PlaneWaveSource
	bl_label = "Plane Wave Source"
	
	####################
	# - Sockets
	####################
	input_sockets = {
		"Temporal Shape": sockets.MaxwellTemporalShapeSocketDef(),
		"Center": sockets.PhysicalPoint3DSocketDef(),
		"Direction": sockets.BoolSocketDef(
			default_value=True,
		),
		"Pol": sockets.PhysicalPolSocketDef(),
	}
	output_sockets = {
		"Source": sockets.MaxwellSourceSocketDef(),
	}
	
	####################
	# - Properties
	####################
	inj_axis: bpy.props.EnumProperty(
		name="Injection Axis",
		description="Axis to inject plane wave along",
		items=[
			("X", "X", "X-Axis"),
			("Y", "Y", "Y-Axis"),
			("Z", "Z", "Z-Axis"),
		],
		default="Y",
		update=(lambda self, context: self.sync_prop("inj_axis")),
	)
	
	####################
	# - Output Socket Computation
	####################
	@base.computes_output_socket(
		"Source",
		input_sockets={"Temporal Shape", "Center", "Direction", "Pol"},
		props={"inj_axis"},
	)
	def compute_source(self, input_sockets: dict, props: dict):
		temporal_shape = input_sockets["Temporal Shape"]
		_center = input_sockets["Center"]
		_direction = input_sockets["Direction"]
		_inj_axis = props["inj_axis"]
		pol = input_sockets["Pol"]
		
		direction = {
			False: "-",
			True: "+",
		}[_direction]
		center = tuple(spu.convert_to(_center, spu.um) / spu.um)
		size = {
			"X": (0, math.inf, math.inf),
			"Y": (math.inf, 0, math.inf),
			"Z": (math.inf, math.inf, 0),
		}[_inj_axis]
		
		S0, S1, S2, S3 = tuple(pol)
		
		chi = 0.5 * sp.atan2(S2, S1)
		psi = 0.5 * sp.asin(S3/S0)
		## chi: Pol angle
		## psi: Ellipticity
		
		## TODO: Something's wonky.
		#angle_theta = chi
		#angle_phi = psi
		pol_angle = sp.pi/2 - chi

		# Display the results
		return td.PlaneWave(
			center=tuple(_center),
			size=size,
			source_time=temporal_shape,
			direction="+" if _direction else "-",
			#angle_theta=angle_theta,
			#angle_phi=angle_phi,
			#pol_angle=pol_angle,
		)



####################
# - Blender Registration
####################
BL_REGISTER = [
	PlaneWaveSourceNode,
]
BL_NODES = {
	ct.NodeType.PlaneWaveSource: (
		ct.NodeCategory.MAXWELLSIM_SOURCES
	)
}
