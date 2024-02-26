import typing as typ
import json
from pathlib import Path

import bpy
import sympy as sp
import pydantic as pyd
import tidy3d as td

from .... import contracts
from .... import sockets
from ... import base

INTERNAL_GEONODES = {
	
}

####################
# - Node
####################
class Viewer3DNode(base.MaxwellSimTreeNode):
	node_type = contracts.NodeType.Viewer3D
	
	bl_label = "3D Viewer"
	
	input_sockets = {
		"data": sockets.AnySocketDef(
			label="Data",
		),
	}
	output_sockets = {}
	
	####################
	# - Update
	####################
	def update_cb(self):
		pass


####################
# - Blender Registration
####################
BL_REGISTER = [
	Viewer3DNode,
]
BL_NODES = {
	contracts.NodeType.Viewer3D: (
		contracts.NodeCategory.MAXWELLSIM_OUTPUTS_VIEWERS
	)
}
