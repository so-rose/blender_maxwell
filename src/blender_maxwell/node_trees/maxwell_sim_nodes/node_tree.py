import contextlib
import typing as typ

import bpy

from blender_maxwell.utils import logger

from . import contracts as ct

log = logger.get(__name__)

####################
# - Cache Management
####################
MemAddr = int


class DeltaNodeLinkCache(typ.TypedDict):
	"""Describes change in the `NodeLink`s of a node tree.

	Attributes:
		added: Set of pointers to added node tree links.
		removed: Set of pointers to removed node tree links.
	"""

	added: set[MemAddr]
	removed: set[MemAddr]


class NodeLinkCache:
	"""A pointer-based cache of node links in a node tree.

	Attributes:
		_node_tree: Reference to the owning node tree.
		link_ptrs_as_links:
		link_ptrs: Pointers (as in integer memory adresses) to `NodeLink`s.
		link_ptrs_as_links: Map from pointers to actual `NodeLink`s.
		link_ptrs_from_sockets: Map from pointers to `NodeSocket`s, representing the source of each `NodeLink`.
		link_ptrs_from_sockets: Map from pointers to `NodeSocket`s, representing the destination of each `NodeLink`.
	"""

	def __init__(self, node_tree: bpy.types.NodeTree):
		"""Initialize the cache from a node tree.

		Parameters:
			node_tree: The Blender node tree whose `NodeLink`s will be cached.
		"""
		self._node_tree = node_tree

		# Link PTR and PTR->REF
		self.link_ptrs: set[MemAddr] = set()
		self.link_ptrs_as_links: dict[MemAddr, bpy.types.NodeLink] = {}

		# Socket PTR and PTR->REF
		self.socket_ptrs: set[MemAddr] = set()
		self.socket_ptrs_as_sockets: dict[MemAddr, bpy.types.NodeSocket] = {}
		self.socket_ptr_refcount: dict[MemAddr, int] = {}

		# Link PTR -> Socket PTR
		self.link_ptrs_as_from_socket_ptrs: dict[MemAddr, MemAddr] = {}
		self.link_ptrs_as_to_socket_ptrs: dict[MemAddr, MemAddr] = {}

		# Fill Cache
		self.regenerate()

	def remove_link(self, link_ptr: MemAddr) -> None:
		"""Removes a link pointer from the cache, indicating that the link doesn't exist anymore.

		Notes:
			- **DOES NOT** remove PTR->REF dictionary entries
			- Invoking this method directly causes the removed node links to not be reported as "removed" by `NodeLinkCache.regenerate()`.
			- This **must** be done whenever a node link is deleted.
			- Failure to do so may result in a segmentation fault at arbitrary future time.

		Parameters:
			link_ptr: Pointer to remove from the cache.
		"""
		self.link_ptrs.remove(link_ptr)
		self.link_ptrs_as_links.pop(link_ptr)

	def remove_sockets_by_link_ptr(self, link_ptr: MemAddr) -> None:
		"""Removes a single pointer's reference to its from/to sockets."""
		from_socket_ptr = self.link_ptrs_as_from_socket_ptrs.pop(link_ptr, None)
		to_socket_ptr = self.link_ptrs_as_to_socket_ptrs.pop(link_ptr, None)

		for socket_ptr in [from_socket_ptr, to_socket_ptr]:
			if socket_ptr is None:
				continue

			# Delete w/RefCount Respect
			if self.socket_ptr_refcount[socket_ptr] == 1:
				self.socket_ptrs.remove(socket_ptr)
				self.socket_ptrs_as_sockets.pop(socket_ptr)
				self.socket_ptr_refcount.pop(socket_ptr)
			else:
				self.socket_ptr_refcount[socket_ptr] -= 1

	def regenerate(self) -> DeltaNodeLinkCache:
		"""Regenerates the cache from the internally-linked node tree.

		Notes:
			- This is designed to run within the `update()` invocation of the node tree.
			- This should be a very fast function, since it is called so much.
		"""
		# Compute All NodeLink Pointers
		all_link_ptrs_as_links = {
			link.as_pointer(): link for link in self._node_tree.links
		}
		all_link_ptrs = set(all_link_ptrs_as_links.keys())

		# Compute Added/Removed Links
		added_link_ptrs = all_link_ptrs - self.link_ptrs
		removed_link_ptrs = self.link_ptrs - all_link_ptrs

		# Edge Case: 'from_socket' Reassignment
		## (Reverse engineered) When all:
		##     - Created a new link between the same two nodes.
		##     - Matching 'to_socket'.
		##     - Non-matching 'from_socket' on the same node.
		## -> THEN the link_ptr will not change, but the from_socket ptr should.
		if len(added_link_ptrs) == 0 and len(removed_link_ptrs) == 0:
			# Find the Link w/Reassigned 'from_socket' PTR
			## A bit of a performance hit from the search, but it's an edge case.
			_link_ptr_as_from_socket_ptrs = {
				link_ptr: (
					from_socket_ptr,
					all_link_ptrs_as_links[link_ptr].from_socket.as_pointer(),
				)
				for link_ptr, from_socket_ptr in self.link_ptrs_as_from_socket_ptrs.items()
				if all_link_ptrs_as_links[link_ptr].from_socket.as_pointer()
				!= from_socket_ptr
			}

			# Completely Remove the Old Link (w/Reassigned 'from_socket')
			## This effectively reclassifies the edge case as a normal 're-add'.
			for link_ptr in _link_ptr_as_from_socket_ptrs:
				log.info(
					'Edge-Case - "from_socket" Reassigned in NodeLink w/o New NodeLink Pointer: %s',
					link_ptr,
				)
				self.remove_link(link_ptr)
				self.remove_sockets_by_link_ptr(link_ptr)

			# Recompute Added/Removed Links
			## The algorithm will now detect an "added link".
			added_link_ptrs = all_link_ptrs - self.link_ptrs
			removed_link_ptrs = self.link_ptrs - all_link_ptrs

		# Shuffle Cache based on Change in Links
		## Remove Entries for Removed Pointers
		for removed_link_ptr in removed_link_ptrs:
			self.remove_link(removed_link_ptr)
			## User must manually call 'remove_socket_by_link_ptr' later.
			## For now, leave dangling socket information by-link.

		# Add New Link Pointers
		self.link_ptrs |= added_link_ptrs
		for link_ptr in added_link_ptrs:
			# Add Link PTR->REF
			new_link = all_link_ptrs_as_links[link_ptr]
			self.link_ptrs_as_links[link_ptr] = new_link

			# Retrieve Link Socket Information
			from_socket = new_link.from_socket
			from_socket_ptr = from_socket.as_pointer()
			to_socket = new_link.to_socket
			to_socket_ptr = to_socket.as_pointer()

			# Add Socket PTR, PTR -> REF
			for socket_ptr, bl_socket in zip(  # noqa: B905
				[from_socket_ptr, to_socket_ptr],
				[from_socket, to_socket],
			):
				# Increment RefCount of Socket PTR
				## This happens if another link also uses the same socket.
				## 1. An output socket links to several inputs.
				## 2. A multi-input socket links from several inputs.
				if socket_ptr in self.socket_ptr_refcount:
					self.socket_ptr_refcount[socket_ptr] += 1
				else:
					## RefCount == 0: Add PTR, PTR -> REF
					self.socket_ptrs.add(socket_ptr)
					self.socket_ptrs_as_sockets[socket_ptr] = bl_socket
					self.socket_ptr_refcount[socket_ptr] = 1

			# Add Link PTR -> Socket PTR
			self.link_ptrs_as_from_socket_ptrs[link_ptr] = from_socket_ptr
			self.link_ptrs_as_to_socket_ptrs[link_ptr] = to_socket_ptr

		return {'added': added_link_ptrs, 'removed': removed_link_ptrs}


####################
# - Node Tree Definition
####################
class MaxwellSimTree(bpy.types.NodeTree):
	bl_idname = ct.TreeType.MaxwellSim.value
	bl_label = 'Maxwell Sim Editor'
	bl_icon = ct.Icon.SimNodeEditor

	####################
	# - Lock Methods
	####################
	def unlock_all(self) -> None:
		"""Unlock all nodes in the node tree, making them editable."""
		log.info('Unlocking All Nodes in NodeTree "%s"', self.bl_label)
		for node in self.nodes:
			node.locked = False
			for bl_socket in [*node.inputs, *node.outputs]:
				bl_socket.locked = False

	@contextlib.contextmanager
	def repreview_all(self) -> None:
		all_nodes_with_preview_active = {
			node.instance_id: node for node in self.nodes if node.preview_active
		}
		self.is_currently_repreviewing = True
		self.newly_previewed_nodes = {}

		try:
			yield
		finally:
			for dangling_previewed_node in [
				node
				for node_instance_id, node in all_nodes_with_preview_active.items()
				if node_instance_id not in self.newly_previewed_nodes
			]:
				# log.debug(
				# 'Removing Dangling Preview of Node "{%s}"',
				# str(dangling_previewed_node),
				# )
				dangling_previewed_node.preview_active = False

	def report_show_preview(self, node: bpy.types.Node) -> None:
		if (
			hasattr(self, 'is_currently_repreviewing')
			and self.is_currently_repreviewing
		):
			self.newly_previewed_nodes[node.instance_id] = node

	####################
	# - Init Methods
	####################
	def on_load(self):
		"""Run by Blender when loading the NodeSimTree, ex. on file load, on creation, etc. .

		It's a bit of a "fake" function - in practicality, it's triggered on the first update() function.
		"""
		if hasattr(self, 'node_link_cache'):
			self.node_link_cache.regenerate()
		else:
			self.node_link_cache = NodeLinkCache(self)

	####################
	# - Update Methods
	####################
	def on_node_removed(self, node: bpy.types.Node):
		"""Run by `MaxwellSimNode.free()` when a node is being removed.

		ONLY input socket links are removed from the NodeLink cache.
		- `self.update()` handles link-removal from existing nodes.
		- `self.update()` can't handle link-removal

		Removes node input links from the internal cache (so we don't attempt to update non-existant sockets).
		"""
		## ONLY Input Socket Links are Removed from the NodeLink Cache
		## - update() handles link-removal from still-existing node just fine.
		## - update() does NOT handle link-removal of non-existant nodes.
		for bl_socket in list(node.inputs.values()) + list(node.outputs.values()):
			# Compute About-To-Be-Freed Link Ptrs
			link_ptrs = {link.as_pointer() for link in bl_socket.links}

			if link_ptrs:
				for link_ptr in link_ptrs:
					self.node_link_cache.remove_link(link_ptr)
					self.node_link_cache.remove_sockets_by_link_ptr(link_ptr)

	def on_node_socket_removed(self, bl_socket: bpy.types.NodeSocket) -> None:
		"""Run by `MaxwellSimNode._prune_inactive_sockets()` when a socket is being removed (but not the node).

		Parameters:
			bl_socket: The node socket that's about to be removed.
		"""
		# Compute About-To-Be-Freed Link Ptrs
		link_ptrs = {link.as_pointer() for link in bl_socket.links}

		if link_ptrs:
			for link_ptr in link_ptrs:
				self.node_link_cache.remove_link(link_ptr)
				self.node_link_cache.remove_sockets_by_link_ptr(link_ptr)

	def update(self) -> None:
		"""Monitors all changes to the node tree, potentially responding with appropriate callbacks.

		Notes:
			- Run by Blender when "anything" changes in the node tree.
			- Responds to node link changes with callbacks, with the help of a performant node link cache.
		"""
		if not hasattr(self, 'ignore_update'):
			self.ignore_update = False

		if not hasattr(self, 'node_link_cache'):
			self.on_load()
			## We presume update() is run before the first link is altered.
			## - Else, the first link of the session will not update caches.
			## - We remain slightly unsure of the semantics.
			## - Therefore, self.on_load() is also called as a load_post handler.
			return

		# Ignore Update
		## Manually set to implement link corrections w/o recursion.
		if self.ignore_update:
			return

		# Compute Changes to Node Links
		delta_links = self.node_link_cache.regenerate()

		link_corrections = {
			'to_remove': [],
			'to_add': [],
		}
		for link_ptr in delta_links['removed']:
			# Retrieve Link PTR -> From/To Socket PTR
			## We don't know if they exist yet.
			from_socket_ptr = self.node_link_cache.link_ptrs_as_from_socket_ptrs[
				link_ptr
			]
			to_socket_ptr = self.node_link_cache.link_ptrs_as_to_socket_ptrs[link_ptr]

			# Check Existance of From/To Socket
			## `Node.free()` must report removed sockets, so this here works.
			## If Both Exist: 'to_socket' may "non-consent" to the link removal.
			if (
				from_socket_ptr in self.node_link_cache.socket_ptrs
				and to_socket_ptr in self.node_link_cache.socket_ptrs
			):
				# Retrieve 'from_socket'/'to_socket' REF
				from_socket = self.node_link_cache.socket_ptrs_as_sockets[
					from_socket_ptr
				]
				to_socket = self.node_link_cache.socket_ptrs_as_sockets[to_socket_ptr]

				# Ask 'to_socket' for Consent to Remove Link
				## The link has already been removed, but we can fix that.
				## If NO: Queue re-adding the link (safe since the sockets exist)
				## TODO: Crash if deleting removing linked loose sockets.
				consent_removal = to_socket.allow_remove_link(from_socket)
				if not consent_removal:
					link_corrections['to_add'].append((from_socket, to_socket))
				else:
					to_socket.on_link_removed(from_socket)

			# Ensure Removal of Socket PTRs, PTRs->REFs
			self.node_link_cache.remove_sockets_by_link_ptr(link_ptr)

		for link_ptr in delta_links['added']:
			# Retrieve Link Reference
			link = self.node_link_cache.link_ptrs_as_links[link_ptr]

			# Ask 'to_socket' for Consent to Add Link
			## The link has already been added, but we can fix that.
			## If NO: Queue re-adding the link (safe since the sockets exist)
			consent_added = link.to_socket.allow_add_link(link)
			if not consent_added:
				link_corrections['to_remove'].append(link)
			else:
				link.to_socket.on_link_added(link)

		# Link Corrections
		## ADD: Links that 'to_socket' don't want removed.
		## REMOVE: Links that 'to_socket' don't want added.
		## NOTE: Both remove() and new() recursively triggers update().
		for link in link_corrections['to_remove']:
			self.ignore_update = True
			self.links.remove(link)  ## Recursively triggers update()
			self.ignore_update = False
		for from_socket, to_socket in link_corrections['to_add']:
			## 'to_socket' and 'from_socket' are guaranteed to exist.
			self.ignore_update = True
			self.links.new(from_socket, to_socket)
			self.ignore_update = False

		# Regenerate on Corrections
		## Prevents next update() from trying to correct the corrections.
		## We must remember to trigger '.remove_sockets_by_link_ptr'
		if link_corrections['to_remove'] or link_corrections['to_add']:
			delta_links = self.node_link_cache.regenerate()
			for link_ptr in delta_links['removed']:
				self.node_link_cache.remove_sockets_by_link_ptr(link_ptr)


####################
# - Post-Load Handler
####################
def initialize_sim_tree_node_link_cache(_: bpy.types.Scene):
	"""Whenever a file is loaded, create/regenerate the NodeLinkCache in all trees."""
	for node_tree in bpy.data.node_groups:
		if node_tree.bl_idname == 'MaxwellSimTree':
			node_tree.on_load()


####################
# - Blender Registration
####################
bpy.app.handlers.load_post.append(initialize_sim_tree_node_link_cache)
## TODO: Move to top-level registration.

BL_REGISTER = [
	MaxwellSimTree,
]
