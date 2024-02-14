import tidy3d as td
import sympy as sp
import sympy.physics.units as spu

from .... import contracts
from .... import sockets
from ... import base

class PointDipoleSourceNode(base.MaxwellSimTreeNode):
	node_type = contracts.NodeType.PointDipoleSource
	
	bl_label = "Point Dipole Source"
	#bl_icon = ...
	
	####################
	# - Sockets
	####################
	input_sockets = {
		"center_x": sockets.RealNumberSocketDef(
			label="Center X",
			default_value=0.0,
		),
		"center_y": sockets.RealNumberSocketDef(
			label="Center Y",
			default_value=0.0,
		),
		"center_z": sockets.RealNumberSocketDef(
			label="Center Z",
			default_value=0.0,
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
	@base.computes_output_socket("source", td.PointDipole)
	def compute_source(self: contracts.NodeTypeProtocol) -> td.PointDipole:
		center = (
			self.compute_input("center_x"),
			self.compute_input("center_y"),
			self.compute_input("center_z"),
		)
		
		cheating_pulse = td.GaussianPulse(freq0=200e12, fwidth=20e12)
		
		return td.PointDipole(
			center=center,
			source_time=cheating_pulse,
			interpolate=True,
			polarization="Ex",
		)



####################
# - Blender Registration
####################
BL_REGISTER = [
	PointDipoleSourceNode,
]
BL_NODES = {
	contracts.NodeType.PointDipoleSource: (
		contracts.NodeCategory.MAXWELL_SIM_SOURCES_MODELLED
	)
}
