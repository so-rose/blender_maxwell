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
BL_KMI_REGISTER = [
	*operators.BL_KMI_REGISTER,
]
BL_NODE_CATEGORIES = [
	*node_trees.BL_NODE_CATEGORIES,
]

km = bpy.context.window_manager.keyconfigs.addon.keymaps.new(
	name='Node Editor',
	space_type="NODE_EDITOR",
)
REGISTERED_KEYMAPS = []
def register():
	global REGISTERED_KEYMAPS
	
	for cls in BL_REGISTER:
		bpy.utils.register_class(cls)
	
	for kmi_def in BL_KMI_REGISTER:
		kmi = km.keymap_items.new(
			*kmi_def["_"],
			ctrl=kmi_def["ctrl"],
			shift=kmi_def["shift"],
			alt=kmi_def["alt"],
		)
		REGISTERED_KEYMAPS.append(kmi)
    
def unregister():
	for cls in reversed(BL_REGISTER):
		bpy.utils.unregister_class(cls)
	
	for kmi in REGISTERED_KEYMAPS:
		km.keymap_items.remove(kmi)

if __name__ == "__main__":
	register()
