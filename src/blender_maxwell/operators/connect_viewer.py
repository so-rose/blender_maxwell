import bpy

from blender_maxwell import contracts as ct
from blender_maxwell.utils import logger

log = logger.get(__name__)


class ConnectViewerNode(bpy.types.Operator):
	bl_idname = ct.OperatorType.ConnectViewerNode
	bl_label = 'Connect Viewer to Active'
	bl_description = 'Connect active node to Viewer Node'
	bl_options = {'REGISTER', 'UNDO'}

	@classmethod
	def poll(cls, context):
		space = context.space_data
		return (
			space.type == 'NODE_EDITOR'
			and space.node_tree is not None
			and space.node_tree.bl_idname == 'MaxwellSimTreeType'
		)

	def invoke(self, context, event):
		node_tree = context.space_data.node_tree
		mlocx = event.mouse_region_x
		mlocy = event.mouse_region_y
		bpy.ops.node.select(
			extend=False,
			location=(mlocx, mlocy),
		)
		select_node = context.selected_nodes[0]

		for node in node_tree.nodes:
			if node.bl_idname == 'ViewerNodeType':
				viewer_node = node
				break
		else:
			viewer_node = node_tree.nodes.new('ViewerNodeType')
			viewer_node.location.x = select_node.location.x + 250
			viewer_node.location.y = select_node.location.y
			select_node.select = False

		new_link = True
		for link in viewer_node.inputs[0].links:
			if link.from_node.name == select_node.name:
				new_link = False
				continue
			node_tree.links.remove(link)

		if new_link:
			node_tree.links.new(select_node.outputs[0], viewer_node.inputs[0])
		return {'FINISHED'}


####################
# - Blender Registration
####################
BL_REGISTER = [
	ConnectViewerNode,
]

BL_HOTKEYS = [
	{
		'_': (
			ct.OperatorType.ConnectViewerNode,
			'LEFTMOUSE',
			'PRESS',
		),
		'ctrl': True,
		'shift': True,
		'alt': False,
	},
]
