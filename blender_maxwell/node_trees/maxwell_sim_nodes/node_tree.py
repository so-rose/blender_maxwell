import bpy

from . import contracts

ICON_SIM_TREE = 'MOD_SIMPLEDEFORM'


class BLENDER_MAXWELL_PT_MaxwellSimTreePanel(bpy.types.Panel):
    bl_label = "Node Tree Custom Prop"
    bl_idname = "NODE_PT_custom_prop"
    bl_space_type = 'NODE_EDITOR'
    bl_region_type = 'UI'
    bl_category = 'Item'

    @classmethod
    def poll(cls, context):
        return context.space_data.tree_type == contracts.TreeType.MaxwellSim.value

    def draw(self, context):
        layout = self.layout
        node_tree = context.space_data.node_tree
        
        layout.prop(node_tree, "preview_collection")
        layout.prop(node_tree, "non_preview_collection")

####################
# - Node Tree Definition
####################
class MaxwellSimTree(bpy.types.NodeTree):
	bl_idname = contracts.TreeType.MaxwellSim
	bl_label = "Maxwell Sim Editor"
	bl_icon = contracts.Icon.MaxwellSimTree
	
	preview_collection: bpy.props.PointerProperty(
		name="Preview Collection",
		description="Collection of Blender objects that will be previewed",
		type=bpy.types.Collection,
		update=(lambda self, context: self.trigger_updates())
	)
	non_preview_collection: bpy.props.PointerProperty(
		name="Non-Preview Collection",
		description="Collection of Blender objects that will NOT be previewed",
		type=bpy.types.Collection,
		update=(lambda self, context: self.trigger_updates())
	)
	
	def trigger_updates(self):
		pass

####################
# - Blender Registration
####################
BL_REGISTER = [
	MaxwellSimTree,
]



####################
# - Red Edges on Error
####################
## TODO: Refactor
#def link_callback_new(context):
#	print("A THING HAPPENED")
#	node_tree_type = contracts.TreeType.MaxwellSim.value
#	link = context.link
#	
#	if not (
#		link.from_node.node_tree.bl_idname == node_tree_type
#		and link.to_node.node_tree.bl_idname == node_tree_type
#	):
#		return
#	
#	source_node = link.from_node
#	
#	source_socket_name = source_node.g_output_socket_name(
#		link.from_socket.name
#	)
#	link_data = source_node.compute_output(source_socket_name)
#	
#	destination_socket = link.to_socket
#	link.is_valid = destination_socket.is_compatible(link_data)
#	
#	print(source_node, destination_socket, link.is_valid)
#
#bpy.msgbus.subscribe_rna(
#	key=("active", "node_tree"),
#	owner=MaxwellSimTree,
#	args=(bpy.context,),
#	notify=link_callback_new,
#	options={'PERSISTENT'}
#)
