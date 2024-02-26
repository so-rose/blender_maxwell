import typing as typ

import bpy
import pydantic as pyd

from .. import base
from ... import contracts

####################
# - Blender Socket
####################
class BlenderMaxwellCreateAndAssignBLObject(bpy.types.Operator):
	bl_idname = "blender_maxwell.create_and_assign_bl_object"
	bl_label = "Create and Assign BL Object"
	
	def execute(self, context):
		mesh = bpy.data.meshes.new("GenMesh")
		new_bl_object = bpy.data.objects.new("GenObj", mesh)
		
		context.collection.objects.link(new_bl_object)
		
		node = context.node
		for bl_socket_name, bl_socket in node.inputs.items():
			if isinstance(bl_socket, BlenderObjectBLSocket):
				bl_socket.default_value = new_bl_object
		
		if hasattr(node, "update_sockets_from_geonodes"):
			node.update_sockets_from_geonodes()
		
		return {'FINISHED'}

####################
# - Blender Socket
####################
class BlenderObjectBLSocket(base.BLSocket):
	socket_type = contracts.SocketType.BlenderObject
	bl_label = "BlenderObject"
	
	####################
	# - Properties
	####################
	raw_value: bpy.props.PointerProperty(
		name="Blender Object",
		description="Represents a Blender object",
		type=bpy.types.Object,
		update=(lambda self, context: self.trigger_updates()),
	)
	
	####################
	# - UI
	####################
	def draw_label_row(self, label_col_row, text):
		label_col_row.label(text=text)
		label_col_row.operator(
			"blender_maxwell.create_and_assign_bl_object",
			text="",
			icon="ADD",
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
class BlenderObjectSocketDef(pyd.BaseModel):
	socket_type: contracts.SocketType = contracts.SocketType.BlenderObject
	label: str
	
	def init(self, bl_socket: BlenderObjectBLSocket) -> None:
		pass

####################
# - Blender Registration
####################
BL_REGISTER = [
	BlenderMaxwellCreateAndAssignBLObject,
	BlenderObjectBLSocket,
]
