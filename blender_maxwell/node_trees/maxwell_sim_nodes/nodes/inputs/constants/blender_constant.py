import typing as typ

from .... import contracts
from .... import sockets
from ... import base

class BlenderConstantNode(base.MaxwellSimTreeNode):
	node_type = contracts.NodeType.BlenderConstant
	
	bl_label = "Blender Constant"
	#bl_icon = constants.ICON_SIM_INPUT
	
	input_sockets = {}
	input_socket_sets = {
		"object": {
			"value": sockets.BlenderObjectSocketDef(
				label="Object",
			),
		},
		"collection": {
			"value": sockets.BlenderCollectionSocketDef(
				label="Collection",
			),
		},
		"image": {
			"value": sockets.BlenderImageSocketDef(
				label="Image",
			),
		},
		"volume": {
			"value": sockets.BlenderVolumeSocketDef(
				label="Volume",
			),
		},
		"text": {
			"value": sockets.BlenderTextSocketDef(
				label="Text",
			),
		},
		"geonodes": {
			"value": sockets.BlenderGeoNodesSocketDef(
				label="GeoNodes",
			),
		},
	}
	output_sockets = {}
	output_socket_sets = input_socket_sets
	
	####################
	# - Callbacks
	####################
	@base.computes_output_socket("value")
	def compute_value(self: contracts.NodeTypeProtocol) -> typ.Any:
		return self.compute_input("value")



####################
# - Blender Registration
####################
BL_REGISTER = [
	BlenderConstantNode,
]
BL_NODES = {
	contracts.NodeType.BlenderConstant: (
		contracts.NodeCategory.MAXWELLSIM_INPUTS_CONSTANTS
	)
}
