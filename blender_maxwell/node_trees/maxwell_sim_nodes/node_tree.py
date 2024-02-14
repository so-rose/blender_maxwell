import bpy

from . import contracts

ICON_SIM_TREE = 'MOD_SIMPLEDEFORM'

####################
# - Node Tree Definition
####################
class MaxwellSimTree(bpy.types.NodeTree):
	bl_idname = contracts.TreeType.MaxwellSim
	bl_label = "Maxwell Sim Editor"
	bl_icon = contracts.Icon.MaxwellSimTree

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
