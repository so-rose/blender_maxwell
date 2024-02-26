import typing as typ

import bpy
import sympy as sp
import pydantic as pyd

from .. import base
from ... import contracts

def mk_and_assign_target_bl_obj(bl_socket, node, node_tree):
	# Create Mesh and Object
	mesh = bpy.data.meshes.new("Mesh" + bl_socket.node.name)
	new_bl_object = bpy.data.objects.new(bl_socket.node.name, mesh)
	
	# Create Preview Collection and Object
	if bl_socket.show_preview:
		#if not node_tree.preview_collection:
		#	new_collection = bpy.data.collections.new("BLMaxwellPreview")
		#	node_tree.preview_collection = new_collection
		#	
		#	bpy.context.scene.collection.children.link(new_collection)
		
		node_tree.preview_collection.objects.link(new_bl_object)
	
	# Create Non-Preview Collection and Object
	else:
		#if not node_tree.non_preview_collection:
		#	new_collection = bpy.data.collections.new("BLMaxwellNonPreview")
		#	node_tree.non_preview_collection = new_collection
		#	
		#	bpy.context.scene.collection.children.link(new_collection)
		
		node_tree.non_preview_collection.objects.link(new_bl_object)
	
	bl_socket.local_target_object = new_bl_object
	
	if hasattr(node, "update_sockets_from_geonodes"):
		node.update_sockets_from_geonodes()

class BlenderMaxwellCreateAndAssignTargetBLObject(bpy.types.Operator):
	bl_idname = "blender_maxwell.create_and_assign_target_bl_object"
	bl_label = "Create and Assign Target BL Object"
	
	def execute(self, context):
		bl_socket = context.socket
		node = bl_socket.node
		node_tree = node.id_data
		
		mk_and_assign_target_bl_obj(bl_socket, node, node_tree)
		return {'FINISHED'}

####################
# - Blender Socket
####################
class BlenderPreviewTargetBLSocket(base.BLSocket):
	socket_type = contracts.SocketType.BlenderPreviewTarget
	bl_label = "BlenderPreviewTarget"
	
	####################
	# - Properties
	####################
	show_preview: bpy.props.BoolProperty(
		name="Target Object Included in Preview",
		description="Whether or not Blender will preview the target object",
		default=True,
		update=(lambda self, context: self.update_preview()),
	)
	show_definition: bpy.props.BoolProperty(
		name="Show Unit System Definition",
		description="Toggle to show unit system definition",
		default=False,
		update=(lambda self, context: self.trigger_updates()),
	)
	
	
	target_object_pinned: bpy.props.BoolProperty(
		name="Target Object Pinned",
		description="Whether or not Blender will manage the target object",
		default=True,
	)
	preview_collection_pinned: bpy.props.BoolProperty(
		name="Global Preview Collection Pinned",
		description="Whether or not Blender will use the global preview collection",
		default=True,
	)
	non_preview_collection_pinned: bpy.props.BoolProperty(
		name="Global Non-Preview Collection Pinned",
		description="Whether or not Blender will use the global non-preview collection",
		default=True,
	)
	
	
	local_target_object: bpy.props.PointerProperty(
		name="Local Target Blender Object",
		description="Represents a Blender object to apply a preview to",
		type=bpy.types.Object,
		update=(lambda self, context: self.trigger_updates()),
	)
	local_preview_collection: bpy.props.PointerProperty(
		name="Local Preview Collection",
		description="Collection of Blender objects that will be previewed",
		type=bpy.types.Collection,
		update=(lambda self, context: self.trigger_updates())
	)
	local_non_preview_collection: bpy.props.PointerProperty(
		name="Local Non-Preview Collection",
		description="Collection of Blender objects that will NOT be previewed",
		type=bpy.types.Collection,
		update=(lambda self, context: self.trigger_updates())
	)
	
	####################
	# - Methods
	####################
	def update_preview(self):
		node_tree = self.node.id_data
		
		# Target Object Pinned
		if (
			self.show_preview
			and self.local_target_object
			and self.target_object_pinned
		):
			node_tree.non_preview_collection.objects.unlink(self.local_target_object)
			node_tree.preview_collection.objects.link(self.local_target_object)
		
		elif (
			not self.show_preview
			and self.local_target_object
			and self.target_object_pinned
		):
			node_tree.preview_collection.objects.unlink(self.local_target_object)
			node_tree.non_preview_collection.objects.link(self.local_target_object)
		
		# Target Object Not Pinned
		if (
			self.show_preview
			and self.local_target_object
			and not self.target_object_pinned
			and self.local_target_object.name in (
				node_tree.non_preview_collection.objects.keys()
			)
		):
			node_tree.non_preview_collection.objects.unlink(self.local_target_object)
			node_tree.preview_collection.objects.link(self.local_target_object)
		elif (
			not self.show_preview
			and self.local_target_object
			and not self.target_object_pinned
			and self.local_target_object.name in (
				node_tree.preview_collection.objects.keys()
			)
		):
			node_tree.preview_collection.objects.unlink(self.local_target_object)
			node_tree.non_preview_collection.objects.link(self.local_target_object)
		
		self.trigger_updates()
	
	####################
	# - UI
	####################
	def draw_label_row(self, label_col_row: bpy.types.UILayout, text) -> None:
		label_col_row.label(text=text)
		label_col_row.prop(self, "show_preview", toggle=True, text="", icon="SEQ_PREVIEW")
		label_col_row.prop(self, "show_definition", toggle=True, text="", icon="MOD_LENGTH")
	
	def draw_value(self, col: bpy.types.UILayout) -> None:
		node_tree = self.node.id_data
		
		if self.show_definition:
			col_row = col.row(align=True)
			col_row.alignment = "EXPAND"
			col_row.label(text="Target", icon="OBJECT_DATA")
			col_row.prop(
				self,
				"target_object_pinned",
				toggle=True,
				icon="EVENT_A",
				icon_only=True,
			)
			#col_row.operator(
			#	"blender_maxwell.create_and_assign_target_bl_object",
			#	text="",
			#	icon="ADD",
			#)
			
			col_row=col.row(align=True)
			col_row.alignment = "EXPAND"
			if not self.target_object_pinned:
				col_row.prop(self, "local_target_object", text="")
			
			# Non-Preview Collection
			col_row=col.row(align=True)
			col_row.alignment = "EXPAND"
			col_row.label(text="Enabled", icon="COLLECTION_COLOR_04")
			col_row.prop(
				self,
				"preview_collection_pinned",
				toggle=True,
				icon="PINNED",
				icon_only=True,
			)
			
			col_row=col.row(align=True)
			col_row.alignment = "EXPAND"
			if not self.preview_collection_pinned:
				col_row.prop(self, "local_preview_collection", text="")
			
			# Non-Preview Collection
			col_row=col.row(align=True)
			col_row.alignment = "EXPAND"
			col_row.label(text="Disabled", icon="COLLECTION_COLOR_01")
			col_row.prop(
				self,
				"non_preview_collection_pinned",
				toggle=True,
				icon="PINNED",
				icon_only=True,
			)
			
			col_row=col.row(align=True)
			col_row.alignment = "EXPAND"
			if not self.non_preview_collection_pinned:
				col_row.prop(self, "local_non_preview_collection", text="")
	
	####################
	# - Default Value
	####################
	@property
	def default_value(self) -> bpy.types.Object:
		node_tree = self.node.id_data
		if not self.local_target_object and self.target_object_pinned:
			mk_and_assign_target_bl_obj(self, self.node, node_tree)
			return self.local_target_object
		
		return self.local_target_object
	
	@default_value.setter
	def default_value(self, value: typ.Any) -> None:
		pass
	
	####################
	# - Cleanup
	####################
	def free(self) -> None:
		if self.local_target_object:
			bpy.data.meshes.remove(self.local_target_object.data, do_unlink=True)

####################
# - Socket Configuration
####################
class BlenderPreviewTargetSocketDef(pyd.BaseModel):
	socket_type: contracts.SocketType = contracts.SocketType.BlenderPreviewTarget
	label: str
	
	show_preview: bool = True
	
	def init(self, bl_socket: BlenderPreviewTargetBLSocket) -> None:
		pass
		#bl_socket.show_preview = self.show_preview

####################
# - Blender Registration
####################
BL_REGISTER = [
	BlenderMaxwellCreateAndAssignTargetBLObject,
	BlenderPreviewTargetBLSocket,
]
