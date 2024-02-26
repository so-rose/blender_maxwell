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
		"polarization": sockets.PhysicalPolSocketDef(
			label="Polarization",
		),
		"temporal_shape": sockets.MaxwellTemporalShapeSocketDef(
			label="Temporal Shape",
		),
		"center": sockets.PhysicalPoint3DSocketDef(
			label="Center",
		),
		"interpolate": sockets.BoolSocketDef(
			label="Interpolate",
			default_value=True,
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
		polarization = self.compute_input("polarization")
		temporal_shape = self.compute_input("temporal_shape")
		_center = self.compute_input("center")
		interpolate = self.compute_input("interpolate")
		
		center = tuple(spu.convert_to(_center, spu.um) / spu.um)
		
		return td.PointDipole(
			center=center,
			source_time=temporal_shape,
			interpolate=interpolate,
			polarization=polarization,
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
