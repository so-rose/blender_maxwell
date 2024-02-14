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
		"size_x": sockets.RealNumberSocketDef(
			label="Size X",
			default_value=1.0,
		),
		"size_y": sockets.RealNumberSocketDef(
			label="Size Y",
			default_value=1.0,
		),
		"size_z": sockets.RealNumberSocketDef(
			label="Size Z",
			default_value=1.0,
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
	@base.computes_output_socket("structure", td.Box)
	def compute_simulation(self: contracts.NodeTypeProtocol) -> td.Box:
		medium = self.compute_input("medium")
		center = (
			self.compute_input("center_x"),
			self.compute_input("center_y"),
			self.compute_input("center_z"),
		)
		size = (
			self.compute_input("size_x"),
			self.compute_input("size_y"),
			self.compute_input("size_z"),
		)
		
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
		contracts.NodeCategory.MAXWELL_SIM_STRUCTURES_PRIMITIES
	)
}
