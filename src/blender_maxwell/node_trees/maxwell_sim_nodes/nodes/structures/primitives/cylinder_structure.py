import tidy3d as td
import sympy as sp
import sympy.physics.units as spu

from .... import contracts
from .... import sockets
from ... import base


class CylinderStructureNode(base.MaxwellSimTreeNode):
	node_type = contracts.NodeType.CylinderStructure
	bl_label = 'Cylinder Structure'
	# bl_icon = ...

	####################
	# - Sockets
	####################
	input_sockets = {
		'medium': sockets.MaxwellMediumSocketDef(
			label='Medium',
		),
		'center': sockets.PhysicalPoint3DSocketDef(
			label='Center',
		),
		'radius': sockets.PhysicalLengthSocketDef(
			label='Radius',
		),
		'height': sockets.PhysicalLengthSocketDef(
			label='Height',
		),
	}
	output_sockets = {
		'structure': sockets.MaxwellStructureSocketDef(
			label='Structure',
		),
	}

	####################
	# - Output Socket Computation
	####################
	@base.computes_output_socket('structure')
	def compute_simulation(self: contracts.NodeTypeProtocol) -> td.Box:
		medium = self.compute_input('medium')
		_center = self.compute_input('center')
		_radius = self.compute_input('radius')
		_height = self.compute_input('height')

		center = tuple(spu.convert_to(_center, spu.um) / spu.um)
		radius = spu.convert_to(_radius, spu.um) / spu.um
		height = spu.convert_to(_height, spu.um) / spu.um

		return td.Structure(
			geometry=td.Cylinder(
				radius=radius,
				center=center,
				length=height,
			),
			medium=medium,
		)


####################
# - Blender Registration
####################
BL_REGISTER = [
	CylinderStructureNode,
]
BL_NODES = {
	contracts.NodeType.CylinderStructure: (
		contracts.NodeCategory.MAXWELLSIM_STRUCTURES_PRIMITIVES
	)
}
