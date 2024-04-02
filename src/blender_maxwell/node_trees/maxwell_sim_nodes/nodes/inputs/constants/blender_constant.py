import typing as typ

from .... import contracts as ct
from .... import sockets
from ... import base


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
	@base.computes_output_socket('Value', input_sockets={'Value'})
	def compute_value(self, input_sockets) -> typ.Any:
		return input_sockets['Value']


####################
# - Blender Registration
####################
BL_REGISTER = [
	BlenderConstantNode,
]
BL_NODES = {ct.NodeType.BlenderConstant: (ct.NodeCategory.MAXWELLSIM_INPUTS_CONSTANTS)}
