bl_info = {
	"name": "Maxwell Simulation and Visualization",
	"blender": (4, 0, 2),
	"category": "Node",
	"description": "Custom node trees for defining and visualizing Maxwell simulation.",
	"author": "Sofus Albert Høgsbro Rose",
	"version": (0, 1),
	"wiki_url": "https://git.sofus.io/dtu-courses/bsc_thesis",
	"tracker_url": "https://git.sofus.io/dtu-courses/bsc_thesis/issues",
}

####################
# - sys.path Library Inclusion
####################
import sys
sys.path.insert(0, "/home/sofus/src/college/bsc_ge/thesis/code/.cached-dependencies")
## ^^ Placeholder

####################
# - Module Import
####################
if "bpy" not in locals():
	import bpy
	import nodeitems_utils
	try:
		from . import node_trees
		from . import operators
		from . import preferences
	except ImportError:
		import sys
		sys.path.insert(0, "/home/sofus/src/college/bsc_ge/thesis/code/blender-maxwell")
		import node_trees
		import operators
		import preferences
else:
	import importlib
	
	importlib.reload(node_trees)


####################
# - Registration
####################
BL_REGISTER = [
	*node_trees.BL_REGISTER,
	*operators.BL_REGISTER,
	*preferences.BL_REGISTER,
]
BL_NODE_CATEGORIES = [
	*node_trees.BL_NODE_CATEGORIES,
]

def register():
	for cls in BL_REGISTER:
		bpy.utils.register_class(cls)
	
	for bl_node_category in BL_NODE_CATEGORIES:
		nodeitems_utils.register_node_categories(*bl_node_category)
    
def unregister():
	for bl_node_category in reversed(BL_NODE_CATEGORIES):
		try:
			nodeitems_utils.unregister_node_categories(bl_node_category[0])
		except: pass
	
	for cls in reversed(BL_REGISTER):
		try:
			bpy.utils.unregister_class(cls)
		except: pass

if __name__ == "__main__":
	register()
