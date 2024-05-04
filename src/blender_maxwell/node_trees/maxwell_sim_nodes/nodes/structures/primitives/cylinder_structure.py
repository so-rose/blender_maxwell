# blender_maxwell
# Copyright (C) 2024 blender_maxwell Project Contributors
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import sympy.physics.units as spu
import tidy3d as td

from .... import contracts, sockets
from ... import base, events


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
	@events.computes_output_socket('structure')
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
