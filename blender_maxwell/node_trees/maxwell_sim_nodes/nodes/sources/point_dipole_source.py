import tidy3d as td
import sympy as sp
import sympy.physics.units as spu

from ... import contracts
from ... import sockets
from .. import base

class PointDipoleSourceNode(base.MaxwellSimTreeNode):
	node_type = contracts.NodeType.PointDipoleSource
	
	bl_label = "Point Dipole Source"
	#bl_icon = ...
	
	####################
	# - Sockets
	####################
	input_sockets = {
		#"polarization": sockets.PhysicalPolSocketDef(
		#	label="Polarization",
		#),  ## TODO: Exactly how to go about this...
		"temporal_shape": sockets.MaxwellTemporalShapeSocketDef(
			label="Temporal Shape",
		),
		"center": sockets.PhysicalPoint3DSocketDef(
			label="Center",
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
		center = tuple(spu.convert_to(_center, spu.um) / spu.um)
		
		cheating_pol = "Ex"
		## TODO: Fix
		
		return td.PointDipole(
			center=center,
			source_time=temporal_shape,
			interpolate=True,
			polarization=cheating_pol,
		)



####################
# - Blender Registration
####################
BL_REGISTER = [
	PointDipoleSourceNode,
]
BL_NODES = {
	contracts.NodeType.PointDipoleSource: (
		contracts.NodeCategory.MAXWELLSIM_SOURCES
	)
}
