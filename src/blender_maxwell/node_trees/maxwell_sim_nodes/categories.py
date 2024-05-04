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

## TODO: Refactor this whole horrible module.

import bpy
import nodeitems_utils

from . import contracts as ct
from .nodes import BL_NODES

DYNAMIC_SUBMENU_REGISTRATIONS = []


def mk_node_categories(
	tree,
	syllable_prefix=[],
	# root = True,
):
	global DYNAMIC_SUBMENU_REGISTRATIONS
	items = []

	# Add Node Items
	base_category = ct.NodeCategory['_'.join(syllable_prefix)]
	for node_type, node_category in BL_NODES.items():
		if node_category == base_category:
			items.append(nodeitems_utils.NodeItem(node_type.value))

	# Add Node Sub-Menus
	for syllable, sub_tree in tree.items():
		current_syllable_path = syllable_prefix + [syllable]
		current_category = ct.NodeCategory['_'.join(current_syllable_path)]

		# Build Items for Sub-Categories
		subitems = mk_node_categories(
			sub_tree,
			current_syllable_path,
		)
		if len(subitems) == 0:
			continue

		# Define Dynamic Node Submenu
		def draw_factory(items):
			def draw(self, context):
				for nodeitem_or_submenu in items:
					if isinstance(
						nodeitem_or_submenu,
						nodeitems_utils.NodeItem,
					):
						nodeitem = nodeitem_or_submenu
						op_add_node_cfg = self.layout.operator(
							'node.add_node',
							text=nodeitem.label,
						)
						op_add_node_cfg.type = nodeitem.nodetype
						op_add_node_cfg.use_transform = True
					elif isinstance(nodeitem_or_submenu, str):
						submenu_id = nodeitem_or_submenu
						self.layout.menu(submenu_id)

			return draw

		menu_class = type(
			str(current_category.value),
			(bpy.types.Menu,),
			{
				'bl_idname': current_category.value,
				'bl_label': ct.NODE_CAT_LABELS[current_category],
				'draw': draw_factory(tuple(subitems)),
			},
		)

		# Report to Items and Registration List
		items.append(current_category.value)
		DYNAMIC_SUBMENU_REGISTRATIONS.append(menu_class)

	return items


####################
# - Blender Registration
####################
BL_NODE_CATEGORIES = mk_node_categories(
	ct.NodeCategory.get_tree()['MAXWELLSIM'],
	syllable_prefix=['MAXWELLSIM'],
)
BL_REGISTER = [*DYNAMIC_SUBMENU_REGISTRATIONS]  ## Must be run after, right now.


def menu_draw(self, context):
	if context.space_data.tree_type == ct.TreeType.MaxwellSim.value:
		for nodeitem_or_submenu in BL_NODE_CATEGORIES:
			if isinstance(nodeitem_or_submenu, str):
				submenu_id = nodeitem_or_submenu
				self.layout.menu(submenu_id)


bpy.types.NODE_MT_add.append(menu_draw)
