# blender_maxwell
# Copyright (C) 2024 blender_maxwell Project Contributors
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

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
