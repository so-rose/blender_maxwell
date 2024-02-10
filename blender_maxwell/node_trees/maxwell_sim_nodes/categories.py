import bpy
import nodeitems_utils
from . import types
from .nodes import BL_NODES

####################
# - Assembly of Node Categories
####################
class MaxwellSimNodeCategory(nodeitems_utils.NodeCategory):
	@classmethod
	def poll(cls, context):
		"""Constrain node category availability to within a MaxwellSimTree."""
		
		return context.space_data.tree_type == types.TreeType.MaxwellSim.value

DYNAMIC_SUBMENU_REGISTRATIONS = []
def mk_node_categories(
	tree,
	syllable_prefix = [],
	#root = True,
):
	global DYNAMIC_SUBMENU_REGISTRATIONS
	items = []
	
	# Add Node Items
	base_category = types.NodeCategory["_".join(syllable_prefix)]
	for node_type, node_category in BL_NODES.items():
		if node_category == base_category:
			items.append(nodeitems_utils.NodeItem(node_type.value))
	
	# Add Node Sub-Menus
	for syllable, sub_tree in tree.items():
		current_syllable_path = syllable_prefix + [syllable]
		current_category = types.NodeCategory[
			"_".join(current_syllable_path)
		]
		
		# Build Items for Sub-Categories
		subitems = mk_node_categories(
			sub_tree,
			current_syllable_path,
		)
		if len(subitems) == 0: continue
		
		# Define Dynamic Node Submenu
		def draw_factory(items):
			def draw(self, context):
				for nodeitem_or_submenu in items:
					if isinstance(
						nodeitem_or_submenu,
						nodeitems_utils.NodeItem,
					):
						nodeitem = nodeitem_or_submenu
						self.layout.operator(
							"node.add_node",
							text=nodeitem.label,
						).type = nodeitem.nodetype
					elif isinstance(nodeitem_or_submenu, str):
						submenu_id = nodeitem_or_submenu
						self.layout.menu(submenu_id)
			return draw
		
		menu_class = type(current_category.value, (bpy.types.Menu,), {
			'bl_idname': current_category.value,
			'bl_label': types.NodeCategory_to_category_label[current_category],
			'draw': draw_factory(tuple(subitems)),
		})
		
		# Report to Items and Registration List
		items.append(current_category.value)
		DYNAMIC_SUBMENU_REGISTRATIONS.append(menu_class)
	
	return items



####################
# - Blender Registration
####################
BL_NODE_CATEGORIES = mk_node_categories(
	types.NodeCategory.get_tree()["MAXWELL"]["SIM"],
	syllable_prefix = ["MAXWELL", "SIM"],
)
## TODO: refractor, this has a big code smell
BL_REGISTER = [
	*DYNAMIC_SUBMENU_REGISTRATIONS
]  ## Must be run after, right now.

## TEST - TODO this is a big code smell
def menu_draw(self, context):
	if context.space_data.tree_type == types.TreeType.MaxwellSim.value:
		for nodeitem_or_submenu in BL_NODE_CATEGORIES:
			if isinstance(nodeitem_or_submenu, str):
				submenu_id = nodeitem_or_submenu
				self.layout.menu(submenu_id)
	
bpy.types.NODE_MT_add.append(menu_draw)
