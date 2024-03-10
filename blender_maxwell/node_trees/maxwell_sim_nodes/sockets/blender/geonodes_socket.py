import typing as typ

import bpy
import pydantic as pyd

from .. import base
from ... import contracts as ct

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
class BlenderGeoNodesBLSocket(base.MaxwellSimSocket):
	socket_type = ct.SocketType.BlenderGeoNodes
	bl_label = "Geometry Node Tree"
	
	####################
	# - Properties
	####################
	raw_value: bpy.props.PointerProperty(
		name="Blender GeoNodes Tree",
		description="Represents a Blender GeoNodes Tree",
		type=bpy.types.NodeTree,
		poll=(lambda self, obj: obj.bl_idname == 'GeometryNodeTree'),
		update=(lambda self, context: self.sync_prop("raw_value", context)),
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
	# - UI
	####################
	def draw_value(self, col: bpy.types.UILayout) -> None:
		col.prop(self, "raw_value", text="")
	
	####################
	# - Default Value
	####################
	@property
	def value(self) -> bpy.types.NodeTree | None:
		return self.raw_value
	
	@value.setter
	def value(self, value: bpy.types.NodeTree) -> None:
		self.raw_value = value

####################
# - Socket Configuration
####################
class BlenderGeoNodesSocketDef(pyd.BaseModel):
	socket_type: ct.SocketType = ct.SocketType.BlenderGeoNodes
	
	def init(self, bl_socket: BlenderGeoNodesBLSocket) -> None:
		pass

####################
# - Blender Registration
####################
BL_REGISTER = [
	BlenderMaxwellResetGeoNodesSocket,
	BlenderGeoNodesBLSocket,
]
