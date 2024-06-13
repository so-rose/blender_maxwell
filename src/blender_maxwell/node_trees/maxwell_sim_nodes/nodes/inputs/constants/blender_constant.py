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

import typing as typ

from .... import contracts as ct
from .... import sockets
from ... import base, events

FK = ct.FlowKind
FS = ct.FlowSignal


class BlenderConstantNode(base.MaxwellSimNode):
	node_type = ct.NodeType.BlenderConstant
	bl_label = 'Blender Constant'

	input_socket_sets: typ.ClassVar = {
		'Object': {
			'Value': sockets.BlenderObjectSocketDef(),
		},
		'Collection': {
			'Value': sockets.BlenderCollectionSocketDef(),
		},
		'Text': {
			'Value': sockets.BlenderTextSocketDef(),
		},
		'Image': {
			'Value': sockets.BlenderImageSocketDef(),
		},
		'GeoNode Tree': {
			'Value': sockets.BlenderGeoNodesSocketDef(),
		},
	}
	output_socket_sets = input_socket_sets

	####################
	# - Callbacks
	####################
	@events.computes_output_socket('Value', kind=FK.Params, input_sockets={'Value'})
	def compute_value(self, input_sockets) -> typ.Any:
		return input_sockets['Value']


####################
# - Blender Registration
####################
BL_REGISTER = [
	BlenderConstantNode,
]
BL_NODES = {ct.NodeType.BlenderConstant: (ct.NodeCategory.MAXWELLSIM_INPUTS_CONSTANTS)}
