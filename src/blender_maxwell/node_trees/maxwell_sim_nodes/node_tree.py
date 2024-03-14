import typing as typ

import bpy

from . import contracts as ct

####################
# - Cache Management
####################
MemAddr = int

class DeltaNodeLinkCache(typ.TypedDict):
	added: set[MemAddr]
	removed: set[MemAddr]

class NodeLinkCache:
	def __init__(self, node_tree: bpy.types.NodeTree):
		# Initialize Parameters
		self._node_tree = node_tree
		self.link_ptrs_to_links = {}
		self.link_ptrs = set()
		self.link_ptrs_from_sockets = {}
		self.link_ptrs_to_sockets = {}
		
		# Fill Cache
		self.regenerate()
	
	def remove(self, link_ptrs: set[MemAddr]) -> None:
		for link_ptr in link_ptrs:
			self.link_ptrs.remove(link_ptr)
			self.link_ptrs_to_links.pop(link_ptr, None)
		
	def regenerate(self) -> DeltaNodeLinkCache:
		current_link_ptrs_to_links = {
			link.as_pointer(): link for link in self._node_tree.links
		}
		current_link_ptrs = set(current_link_ptrs_to_links.keys())
		
		# Compute Delta
		added_link_ptrs = current_link_ptrs - self.link_ptrs
		removed_link_ptrs = self.link_ptrs - current_link_ptrs
		
		# Update Caches Incrementally
		self.remove(removed_link_ptrs)
		
		self.link_ptrs |= added_link_ptrs
		for link_ptr in added_link_ptrs:
			link = current_link_ptrs_to_links[link_ptr]
			
			self.link_ptrs_to_links[link_ptr] = link
			self.link_ptrs_from_sockets[link_ptr] = link.from_socket
			self.link_ptrs_to_sockets[link_ptr] = link.to_socket
		
		return {"added": added_link_ptrs, "removed": removed_link_ptrs}

####################
# - Node Tree Definition
####################
class MaxwellSimTree(bpy.types.NodeTree):
	bl_idname = ct.TreeType.MaxwellSim.value
	bl_label = "Maxwell Sim Editor"
	bl_icon = ct.Icon.SimNodeEditor.value
	
	####################
	# - Lock Methods
	####################
	def unlock_all(self):
		for node in self.nodes:
			node.locked = False
			for bl_socket in [*node.inputs, *node.outputs]:
				bl_socket.locked = False
	
	####################
	# - Init Methods
	####################
	def on_load(self):
		"""Run by Blender when loading the NodeSimTree, ex. on file load, on creation, etc. .
		
		It's a bit of a "fake" function - in practicality, it's triggered on the first update() function.
		"""
		## TODO: Consider tying this to an "on_load" handler
		self._node_link_cache = NodeLinkCache(self)
		
	
	####################
	# - Update Methods
	####################
	def sync_node_removed(self, node: bpy.types.Node):
		"""Run by `Node.free()` when a node is being removed.
		
		Removes node input links from the internal cache (so we don't attempt to update non-existant sockets).
		"""
		for bl_socket in node.inputs.values():
			# Retrieve Socket Links (if any)
			self._node_link_cache.remove({
				link.as_pointer()
				for link in bl_socket.links
			})
		## ONLY Input Socket Links are Removed from the NodeLink Cache
		## - update() handles link-removal from still-existing node just fine.
		## - update() does NOT handle link-removal of non-existant nodes.
	
	def update(self):
		"""Run by Blender when 'something changes' in the node tree.
		
		Updates an internal node link cache, then updates sockets that just lost/gained an input link.
		"""
		if not hasattr(self, "_node_link_cache"):
			self.on_load()
			## We presume update() is run before the first link is altered.
			## - Else, the first link of the session will not update caches.
			## - We remain slightly unsure of the semantics.
			## - More testing needed to prevent this 'first-link bug'.
			return
		
		# Compute Changes to NodeLink Cache
		delta_links = self._node_link_cache.regenerate()
		
		link_alterations = {
			"to_remove": [],
			"to_add": [],
		}
		for link_ptr in delta_links["removed"]:
			from_socket = self._node_link_cache.link_ptrs_from_sockets[link_ptr]
			to_socket = self._node_link_cache.link_ptrs_to_sockets[link_ptr]
			
			# Update Socket Caches
			self._node_link_cache.link_ptrs_from_sockets.pop(link_ptr, None)
			self._node_link_cache.link_ptrs_to_sockets.pop(link_ptr, None)
			
			# Trigger Report Chain on Socket that Just Lost a Link
			## Aka. Forward-Refresh Caches Relying on Linkage
			if not (
				consent_removal := to_socket.sync_link_removed(from_socket)
			):
				# Did Not Consent to Removal: Queue Add Link
				link_alterations["to_add"].append((from_socket, to_socket))
		
		for link_ptr in delta_links["added"]:
			link = self._node_link_cache.link_ptrs_to_links.get(link_ptr)
			if link is None: continue
			
			# Trigger Report Chain on Socket that Just Gained a Link
			## Aka. Forward-Refresh Caches Relying on Linkage
			
			if not (
				consent_added := link.to_socket.sync_link_added(link)
			):
				# Did Not Consent to Addition: Queue Remove Link
				link_alterations["to_remove"].append(link)
		
		# Execute Queued Operations
		## - Especially undoing undesirable link changes.
		## - This is important for locked graphs, whose links must not change.
		for link in link_alterations["to_remove"]:
			self.links.remove(link)
		for from_socket, to_socket in link_alterations["to_add"]:
			self.links.new(from_socket, to_socket)
		
		# If Queued Operations: Regenerate Cache
		## - This prevents the next update() from picking up on alterations.
		if link_alterations["to_remove"] or link_alterations["to_add"]:
			self._node_link_cache.regenerate()

####################
# - Post-Load Handler
####################
def initialize_sim_tree_node_link_cache(scene: bpy.types.Scene):
	"""Whenever a file is loaded, create/regenerate the NodeLinkCache in all trees.
	"""
	for node_tree in bpy.data.node_groups:
		if node_tree.bl_idname == "MaxwellSimTree":
			if not hasattr(node_tree, "_node_link_cache"):
				node_tree._node_link_cache = NodeLinkCache(node_tree)
			else:
				node_tree._node_link_cache.regenerate()

####################
# - Blender Registration
####################
bpy.app.handlers.load_post.append(initialize_sim_tree_node_link_cache)

BL_REGISTER = [
	MaxwellSimTree,
]
