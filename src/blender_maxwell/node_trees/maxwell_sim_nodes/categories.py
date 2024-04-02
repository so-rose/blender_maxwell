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
## TODO: refactor, this has a big code smell
BL_REGISTER = [*DYNAMIC_SUBMENU_REGISTRATIONS]  ## Must be run after, right now.


## TEST - TODO this is a big code smell
def menu_draw(self, context):
	if context.space_data.tree_type == ct.TreeType.MaxwellSim.value:
		for nodeitem_or_submenu in BL_NODE_CATEGORIES:
			if isinstance(nodeitem_or_submenu, str):
				submenu_id = nodeitem_or_submenu
				self.layout.menu(submenu_id)


bpy.types.NODE_MT_add.append(menu_draw)
