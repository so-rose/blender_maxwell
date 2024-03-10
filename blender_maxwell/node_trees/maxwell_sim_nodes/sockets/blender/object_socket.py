import typing as typ

import bpy
import pydantic as pyd

from .. import base
from ... import contracts as ct

####################
# - Blender Socket
####################
class BlenderMaxwellCreateAndAssignBLObject(bpy.types.Operator):
	bl_idname = "blender_maxwell.create_and_assign_bl_object"
	bl_label = "Create and Assign BL Object"
	
	## TODO: Refactor
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
class BlenderObjectBLSocket(base.MaxwellSimSocket):
	socket_type = ct.SocketType.BlenderObject
	bl_label = "Blender Object"
	
	####################
	# - Properties
	####################
	raw_value: bpy.props.PointerProperty(
		name="Blender Object",
		description="Represents a Blender object",
		type=bpy.types.Object,
		update=(lambda self, context: self.sync_prop("raw_value", context)),
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
	
	def draw_value(self, col: bpy.types.UILayout) -> None:
		col.prop(self, "raw_value", text="")
	
	####################
	# - Default Value
	####################
	@property
	def value(self) -> bpy.types.Object | None:
		return self.raw_value
	
	@value.setter
	def value(self, value: bpy.types.Object) -> None:
		self.raw_value = value

####################
# - Socket Configuration
####################
class BlenderObjectSocketDef(pyd.BaseModel):
	socket_type: ct.SocketType = ct.SocketType.BlenderObject
	
	def init(self, bl_socket: BlenderObjectBLSocket) -> None:
		pass

####################
# - Blender Registration
####################
BL_REGISTER = [
	BlenderMaxwellCreateAndAssignBLObject,
	BlenderObjectBLSocket,
]
