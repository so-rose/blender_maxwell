import math

import tidy3d as td
import sympy as sp
import sympy.physics.units as spu

from ... import contracts
from ... import sockets
from .. import base

class PlaneWaveSourceNode(base.MaxwellSimTreeNode):
	node_type = contracts.NodeType.PlaneWaveSource
	
	bl_label = "Plane Wave Source"
	#bl_icon = ...
	
	####################
	# - Sockets
	####################
	input_sockets = {
		"temporal_shape": sockets.MaxwellTemporalShapeSocketDef(
			label="Temporal Shape",
		),
		"center": sockets.PhysicalPoint3DSocketDef(
			label="Center",
		),
		"size": sockets.PhysicalSize3DSocketDef(
			label="Size",
		),
		"direction": sockets.BoolSocketDef(
			label="+ Direction?",
			default_value=True,
		),
		"angle_theta": sockets.PhysicalAngleSocketDef(
			label="θ",
		),
		"angle_phi": sockets.PhysicalAngleSocketDef(
			label="φ",
		),
		"angle_pol": sockets.PhysicalAngleSocketDef(
			label="Pol Angle",
		),
	}
	output_sockets = {
		"source": sockets.MaxwellSourceSocketDef(
			label="Source",
		),
	}
	
	####################
	# - Output Socket Computation
	####################
	@base.computes_output_socket("source")
	def compute_source(self: contracts.NodeTypeProtocol) -> td.PointDipole:
		temporal_shape = self.compute_input("temporal_shape")
		_center = self.compute_input("center")
		_size = self.compute_input("size")
		_direction = self.compute_input("direction")
		_angle_theta = self.compute_input("angle_theta")
		_angle_phi = self.compute_input("angle_phi")
		_angle_pol = self.compute_input("angle_pol")
		
		center = tuple(spu.convert_to(_center, spu.um) / spu.um)
		size = tuple(
			0 if val == 1.0 else math.inf
			for val in spu.convert_to(_size, spu.um) / spu.um
		)
		angle_theta = spu.convert_to(_angle_theta, spu.rad) / spu.rad
		angle_phi = spu.convert_to(_angle_phi, spu.rad) / spu.rad
		angle_pol = spu.convert_to(_angle_pol, spu.rad) / spu.rad
		
		return td.PlaneWave(
			center=center,
			size=size,
			source_time=temporal_shape,
			direction="+" if _direction else "-",
			angle_theta=angle_theta,
			angle_phi=angle_phi,
			pol_angle=angle_pol,
		)



####################
# - Blender Registration
####################
BL_REGISTER = [
	PlaneWaveSourceNode,
]
BL_NODES = {
	contracts.NodeType.PlaneWaveSource: (
		contracts.NodeCategory.MAXWELLSIM_SOURCES
	)
}
