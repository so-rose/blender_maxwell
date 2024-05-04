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

from ... import contracts as ct
from ... import sockets
from .. import base, events


class UnitSystemConstantNode(base.MaxwellSimNode):
	node_type = ct.NodeType.UnitSystem
	bl_label = 'Unit System'

	input_sockets = {
		'Unit System': sockets.PhysicalUnitSystemSocketDef(
			show_by_default=True,
		),
	}
	output_sockets = {
		'Unit System': sockets.PhysicalUnitSystemSocketDef(),
	}

	####################
	# - Callbacks
	####################
	@events.computes_output_socket(
		'Unit System',
		input_sockets={'Unit System'},
	)
	def compute_unit_system(self, input_sockets) -> dict:
		return input_sockets['Unit System']


####################
# - Blender Registration
####################
BL_REGISTER = [
	UnitSystemConstantNode,
]
BL_NODES = {ct.NodeType.UnitSystem: (ct.NodeCategory.MAXWELLSIM_INPUTS)}
