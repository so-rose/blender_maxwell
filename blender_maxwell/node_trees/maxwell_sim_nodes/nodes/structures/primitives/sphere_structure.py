import tidy3d as td
import sympy as sp
import sympy.physics.units as spu

from .... import contracts
from .... import sockets
from ... import base

class SphereStructureNode(base.MaxwellSimTreeNode):
	node_type = contracts.NodeType.SphereStructure
	bl_label = "Sphere Structure"
	#bl_icon = ...
	
	####################
	# - Sockets
	####################
	input_sockets = {
		"medium": sockets.MaxwellMediumSocketDef(
			label="Medium",
		),
		"center": sockets.PhysicalPoint3DSocketDef(
			label="Center",
		),
		"radius": sockets.PhysicalLengthSocketDef(
			label="Radius",
		),
	}
	output_sockets = {
		"structure": sockets.MaxwellStructureSocketDef(
			label="Structure",
		),
	}
	
	####################
	# - Output Socket Computation
	####################
	@base.computes_output_socket("structure")
	def compute_simulation(self: contracts.NodeTypeProtocol) -> td.Box:
		medium = self.compute_input("medium")
		_center = self.compute_input("center")
		_radius = self.compute_input("radius")
		
		center = tuple(spu.convert_to(_center, spu.um) / spu.um)
		radius = spu.convert_to(_radius, spu.um) / spu.um
		
		return td.Structure(
			geometry=td.Sphere(
				radius=radius,
				center=center,
			),
			medium=medium,
		)



####################
# - Blender Registration
####################
BL_REGISTER = [
	SphereStructureNode,
]
BL_NODES = {
	contracts.NodeType.SphereStructure: (
		contracts.NodeCategory.MAXWELLSIM_STRUCTURES_PRIMITIVES
	)
}
