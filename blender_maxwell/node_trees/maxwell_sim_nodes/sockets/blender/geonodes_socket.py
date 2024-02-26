import typing as typ

import bpy
import pydantic as pyd

from .. import base
from ... import contracts

####################
# - Operators
####################
class BlenderMaxwellResetGeoNodesSocket(bpy.types.Operator):
	bl_idname = "blender_maxwell.reset_geo_nodes_socket"
	bl_label = "Reset GeoNodes Socket"
	
	def execute(self, context):
		context.socket.update_geonodes_node()
		
		return {'FINISHED'}


####################
# - Blender Socket
####################
class BlenderGeoNodesBLSocket(base.BLSocket):
	socket_type = contracts.SocketType.BlenderGeoNodes
	bl_label = "BlenderGeoNodes"
	
	####################
	# - Properties
	####################
	def update_geonodes_node(self):
		if hasattr(self.node, "update_sockets_from_geonodes"):
			self.node.update_sockets_from_geonodes()
		else:
			raise ValueError("Node doesn't have GeoNodes socket update method.")
		
		# Run the Usual Updates
		self.trigger_updates()
		
	raw_value: bpy.props.PointerProperty(
		name="Blender GeoNodes Tree",
		description="Represents a Blender GeoNodes Tree",
		type=bpy.types.NodeTree,
		poll=(lambda self, obj: obj.bl_idname == 'GeometryNodeTree'),
		update=(lambda self, context: self.update_geonodes_node()),
	)
	
	####################
	# - UI
	####################
	def draw_label_row(self, label_col_row, text):
		label_col_row.label(text=text)
		if self.raw_value:
			label_col_row.operator(
				"blender_maxwell.reset_geo_nodes_socket",
				text="",
				icon="FILE_REFRESH",
			)
	
	####################
	# - Default Value
	####################
	@property
	def default_value(self) -> bpy.types.Object | None:
		return self.raw_value
	
	@default_value.setter
	def default_value(self, value: bpy.types.Object) -> None:
		self.raw_value = value

####################
# - Socket Configuration
####################
class BlenderGeoNodesSocketDef(pyd.BaseModel):
	socket_type: contracts.SocketType = contracts.SocketType.BlenderGeoNodes
	label: str
	
	def init(self, bl_socket: BlenderGeoNodesBLSocket) -> None:
		pass

####################
# - Blender Registration
####################
BL_REGISTER = [
	BlenderMaxwellResetGeoNodesSocket,
	BlenderGeoNodesBLSocket,
]
