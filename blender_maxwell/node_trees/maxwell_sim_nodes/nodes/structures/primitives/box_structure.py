import tidy3d as td
import sympy as sp
import sympy.physics.units as spu

from .... import contracts
from .... import sockets
from ... import base

class BoxStructureNode(base.MaxwellSimTreeNode):
	node_type = contracts.NodeType.BoxStructure
	bl_label = "Box Structure"
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
		"size": sockets.PhysicalSize3DSocketDef(
			label="Size",
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
		_size = self.compute_input("size")
		
		center = tuple(spu.convert_to(_center, spu.um) / spu.um)
		size = tuple(spu.convert_to(_size, spu.um) / spu.um)
		
		return td.Structure(
			geometry=td.Box(
				center=center,
				size=size,
			),
			medium=medium,
		)



####################
# - Blender Registration
####################
BL_REGISTER = [
	BoxStructureNode,
]
BL_NODES = {
	contracts.NodeType.BoxStructure: (
		contracts.NodeCategory.MAXWELLSIM_STRUCTURES_PRIMITIVES
	)
}
