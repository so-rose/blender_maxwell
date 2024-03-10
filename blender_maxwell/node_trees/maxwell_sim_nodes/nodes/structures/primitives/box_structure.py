import tidy3d as td
import sympy as sp
import sympy.physics.units as spu

from .... import contracts as ct
from .... import sockets
from ... import base

class BoxStructureNode(base.MaxwellSimNode):
	node_type = ct.NodeType.BoxStructure
	bl_label = "Box Structure"
	
	####################
	# - Sockets
	####################
	input_sockets = {
		"Medium": sockets.MaxwellMediumSocketDef(),
		"Center": sockets.PhysicalPoint3DSocketDef(),
		"Size": sockets.PhysicalSize3DSocketDef(),
	}
	output_sockets = {
		"Structure": sockets.MaxwellStructureSocketDef(),
	}
	
	####################
	# - Output Socket Computation
	####################
	@base.computes_output_socket(
		"Structure",
		input_sockets={"Medium", "Center", "Size"},
	)
	def compute_simulation(self, input_sockets: dict) -> td.Box:
		medium = input_sockets["Medium"]
		_center = input_sockets["Center"]
		_size = input_sockets["Size"]
		
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
	ct.NodeType.BoxStructure: (
		ct.NodeCategory.MAXWELLSIM_STRUCTURES_PRIMITIVES
	)
}
