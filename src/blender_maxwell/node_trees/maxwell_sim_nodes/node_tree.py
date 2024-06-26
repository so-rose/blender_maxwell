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

import contextlib
import functools
import queue
import typing as typ

import bpy

from blender_maxwell.utils import logger, serialize

from . import contracts as ct
from .managed_objs.managed_bl_image import ManagedBLImage

log = logger.get(__name__)

link_action_queue = queue.Queue()


def set_link_validity(link: bpy.types.NodeLink, validity: bool) -> None:
	log.critical('Set %s validity to %s', str(link), str(validity))
	link.is_valid = validity


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
	"""A volatile pointer-based cache of node links in a node tree.

	Warnings:
		Everything here is **extremely** unsafe.
		Even a single mistake **will** cause a use-after-free crash of Blender.

		Used perfectly, it allows for powerful features; anything less, and it's an epic liability.

	Attributes:
		_node_tree: Reference to the node tree for which this cache is valid.
		link_ptrs: Memory-address identifiers for all node links that currently exist in `_node_tree`.
		link_ptrs_as_links: Mapping from pointers (integers) to actual `NodeLink` objects.
			**WARNING**: If the pointer-referenced object no longer exists, then Blender **will crash immediately** upon attempting to use it. There is no way to mitigate this.
		socket_ptrs: Memory-address identifiers for all sockets that currently exist in `_node_tree`.
		socket_ptrs_as_sockets: Mapping from pointers (integers) to actual `NodeSocket` objects.
			**WARNING**: If the pointer-referenced object no longer exists, then Blender **will crash immediately** upon attempting to use it. There is no way to mitigate this.
		socket_ptr_refcount: The amount of links currently connected to a given socket pointer.
			Used to drive the deletion of socket pointers using only knowledge about `link_ptr` removal.
		link_ptrs_as_from_socket_ptrs: The pointer of the source socket, defined for every node link pointer.
		link_ptrs_as_to_socket_ptrs: The pointer of the destination socket, defined for every node link pointer.
	"""

	def __init__(self, node_tree: bpy.types.NodeTree):
		"""Defines and fills the cache from a live node tree."""
		self._node_tree = node_tree

		self.link_ptrs: set[MemAddr] = set()
		self.link_ptrs_as_links: dict[MemAddr, bpy.types.NodeLink] = {}

		self.socket_ptrs: set[MemAddr] = set()
		self.socket_ptrs_as_sockets: dict[MemAddr, bpy.types.NodeSocket] = {}
		self.socket_ptr_refcount: dict[MemAddr, int] = {}

		self.link_ptrs_as_from_socket_ptrs: dict[MemAddr, MemAddr] = {}
		self.link_ptrs_as_to_socket_ptrs: dict[MemAddr, MemAddr] = {}

		self.link_ptrs_invalid: set[MemAddr] = set()

		# Fill Cache
		self.regenerate()

	def remove_link(self, link_ptr: MemAddr) -> None:
		"""Reports a link as removed, causing it to be removed from the cache.

		This **must** be run whenever a node link is deleted.
		**Failure to to so WILL result in segmentation fault** at an unknown future time.

		In particular, the following actions are taken:
		- The entry in `self.link_ptrs_as_links` is deleted.
		- Any entry in `self.link_ptrs_invalid` is deleted (if exists).

		Notes:
			Invoking this method directly causes the removed node links to not be reported as "removed" by `NodeLinkCache.regenerate()`.
			In some cases, this may be desirable, ex. for internal methods that shouldn't trip a `DataChanged` flow event.

		Parameters:
			link_ptr: The pointer (integer) to remove from the cache.

		Raises:
			KeyError: If `link_ptr` is not a member of either `self.link_ptrs`, or of `self.link_ptrs_as_links`.
		"""
		self.link_ptrs.remove(link_ptr)
		self.link_ptrs_as_links.pop(link_ptr)

		if link_ptr in self.link_ptrs_invalid:
			self.link_ptrs_invalid.remove(link_ptr)

	def remove_sockets_by_link_ptr(self, link_ptr: MemAddr) -> None:
		"""Deassociate from all sockets referenced by a link, respecting the socket pointer reference-count.

		The `NodeLinkCache` stores references to all socket pointers referenced by any link.
		Since several links can be associated with each socket, we must keep a "reference count" per-socket.
		When the "reference count" drops to zero, then there are no longer any `NodeLink`s that refer to it, and therefore it should be removed from the `NodeLinkCache`.

		This method facilitates that process by:
		- Extracting (with removal) the from / to socket pointers associated with `link_ptr`.
		- If the socket pointer has a reference count of `1`, then it is **completely removed**.
		- If the socket pointer has a reference count of `>1`, then the reference count is decremented by `1`.

		Notes:
			In general, this should be called together with `remove_link`.
			However, in certain cases, this process also needs to happen by itself.

		Parameters:
			link_ptr: The pointer (integer) to remove from the cache.
		"""
		# Remove Socket Pointers
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
		"""Efficiently scans the internally referenced node tree to thoroughly update all attributes of this `NodeLinkCache`.

		Notes:
			This runs in a **very** hot loop, within the `update()` function of the node tree.
			Anytime anything happens in the node tree, `update()` (and therefore this method) is called.

			Thus, performance is of the utmost importance.
			Just a few microseconds too much may be amplified dozens of times over in practice, causing big stutters.
		"""
		# Compute All NodeLink Pointers
		## -> It can be very inefficient to do any full-scan of the node tree.
		## -> However, simply extracting the pointer: link ends up being fast.
		## -> This pattern seems to be the best we can do, efficiency-wise.
		all_link_ptrs_as_links = {
			link.as_pointer(): link for link in self._node_tree.links
		}
		all_link_ptrs = set(all_link_ptrs_as_links.keys())

		# Compute Added/Removed Links
		## -> In essence, we've created a 'diff' here.
		## -> Set operations are fast, and expressive!
		added_link_ptrs = all_link_ptrs - self.link_ptrs
		removed_link_ptrs = self.link_ptrs - all_link_ptrs

		# Edge Case: 'from_socket' Reassignment
		## (Reverse Engineered) When all are true:
		##     - Created a new link between the same nodes as previous link.
		##     - Matching 'to_socket' as the previous link.
		##     - Non-matching 'from_socket', but on the same node.
		## -> THEN the link_ptr will not change, but the from_socket ptr does.
		if not added_link_ptrs and not removed_link_ptrs:
			# Find the Link w/Reassigned 'from_socket' PTR
			## -> This isn't very fast, but the edge case isn't so common.
			## -> Comprehensions are still quite optimized.
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
			## -> Casts the edge case to look like a typical 're-add'.
			for link_ptr in _link_ptr_as_from_socket_ptrs:
				log.debug(
					'Edge-Case - "from_socket" Reassigned in NodeLink w/o New NodeLink Pointer: %s',
					link_ptr,
				)
				self.remove_link(link_ptr)
				self.remove_sockets_by_link_ptr(link_ptr)

			# Recompute Added/Removed Links
			## -> Guide the usual algorithm to detect an "added link".
			added_link_ptrs = all_link_ptrs - self.link_ptrs
			removed_link_ptrs = self.link_ptrs - all_link_ptrs

		# Delete Removed Links
		## -> NOTE: We leave dangling socket information on purpose.
		## -> This information will be used to ask for 'removal consent'.
		## -> To truly remove, must call 'remove_socket_by_link_ptr' later.
		for removed_link_ptr in removed_link_ptrs:
			self.remove_link(removed_link_ptr)

		# Create Added Links
		## -> First, simply concatenate the added link pointers.
		self.link_ptrs |= added_link_ptrs
		for link_ptr in added_link_ptrs:
			# Create Pointer -> Reference Entry
			## -> This allows us to efficiently access the link by-pointer.
			## -> Doing so otherwise requires a full search.
			## -> **If link is deleted w/o report, access will cause crash**.
			new_link = all_link_ptrs_as_links[link_ptr]
			self.link_ptrs_as_links[link_ptr] = new_link

			# Retrieve Link Socket Information
			from_socket = new_link.from_socket
			from_socket_ptr = from_socket.as_pointer()
			to_socket = new_link.to_socket
			to_socket_ptr = to_socket.as_pointer()

			# Add Socket Information
			for socket_ptr, bl_socket in zip(  # noqa: B905
				[from_socket_ptr, to_socket_ptr],
				[from_socket, to_socket],
			):
				# RefCount > 0: Increment RefCount of Socket PTR
				## This happens if another link also uses the same socket.
				## 1. An output socket links to several inputs.
				## 2. A multi-input socket links from several inputs.
				if socket_ptr in self.socket_ptr_refcount:
					self.socket_ptr_refcount[socket_ptr] += 1

				# RefCount == 0: Create Socket Pointer w/Reference
				## -> Also initialize the refcount for the socket pointer.
				else:
					self.socket_ptrs.add(socket_ptr)
					self.socket_ptrs_as_sockets[socket_ptr] = bl_socket
					self.socket_ptr_refcount[socket_ptr] = 1

			# Add Entry from Link Pointer -> Socket Pointer
			self.link_ptrs_as_from_socket_ptrs[link_ptr] = from_socket_ptr
			self.link_ptrs_as_to_socket_ptrs[link_ptr] = to_socket_ptr

		return {'added': added_link_ptrs, 'removed': removed_link_ptrs}

	def update_validity(self) -> DeltaNodeLinkCache:
		"""Query all cached links to determine whether they are valid."""
		self.link_ptrs_invalid = {
			link_ptr for link_ptr, link in self.link_ptrs_as_links if not link.is_valid
		}

	def report_validity(self, link_ptr: MemAddr, validity: bool) -> None:
		"""Report a link as invalid."""
		if validity and link_ptr in self.link_ptrs_invalid:
			self.link_ptrs_invalid.remove(link_ptr)
		elif not validity and link_ptr not in self.link_ptrs_invalid:
			self.link_ptrs_invalid.add(link_ptr)

	def set_validities(self) -> None:
		"""Set the validity of links in the node tree according to the internal cache.

		Validity doesn't need to be removed, as update() automatically cleans up by default.
		"""
		for link in [
			link
			for link_ptr, link in self.link_ptrs_as_links.items()
			if link_ptr in self.link_ptrs_invalid
		]:
			if link.is_valid:
				link.is_valid = False


####################
# - Node Tree Definition
####################
class MaxwellSimTree(bpy.types.NodeTree):
	"""Node tree containing a node-based program for design and analysis of Maxwell PDE simulations.

	Attributes:
		is_active: Whether the node tree should be considered to be in a usable state, capable of updating Blender data.
			In general, only one `MaxwellSimTree` should be active at a time.
	"""

	bl_idname = ct.TreeType.MaxwellSim.value
	bl_label = 'Maxwell Sim Editor'
	bl_icon = ct.Icon.SimNodeEditor

	is_active: bpy.props.BoolProperty(
		default=True,
	)

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
	# - Lock Methods
	####################
	def unlock_all(self) -> None:
		"""Unlock all nodes in the node tree, making them editable.

		Notes:
			All `MaxwellSimNode`s have a `.locked` attribute, which prevents the entire UI from being modified.

			This method simply sets the `locked` attribute to `False` on all nodes.
		"""
		log.info('Unlocking All Nodes in NodeTree "%s"', self.bl_label)
		for node in self.nodes:
			if node.type in ['REROUTE', 'FRAME']:
				continue

			# Unlock Node
			if node.locked:
				node.locked = False

			# Unlock Node Sockets
			for bl_socket in [*node.inputs, *node.outputs]:
				if bl_socket.locked:
					bl_socket.locked = False

	####################
	# - Link Update Methods
	####################
	def report_link_validity(self, link: bpy.types.NodeLink, validity: bool) -> None:
		"""Report that a particular `NodeLink` should be considered to be either valid or invalid.

		The `NodeLink.is_valid` attribute is generally (and automatically) used to indicate the detection of cycles in the node tree.
		However, visually, it causes a very clear "error red" highlight to appear on the node link, which can extremely useful when determining the reasons behind unexpected outout.

		Notes:
			Run by `MaxwellSimSocket` when a link should be shown to be "invalid".
		"""
		## TODO: Doesn't quite work.
		# log.debug(
		# 'Reported Link Validity %s (is_valid=%s, from_socket=%s, to_socket=%s)',
		# validity,
		# link.is_valid,
		# link.from_socket,
		# link.to_socket,
		# )
		# self.node_link_cache.report_validity(link.as_pointer(), validity)

	####################
	# - Node Update Methods
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

	def update(self) -> None:  # noqa: PLR0912, C901
		"""Monitors all changes to the node tree, potentially responding with appropriate callbacks.

		Notes:
			- Run by Blender when "anything" changes in the node tree.
			- Responds to node link changes with callbacks, with the help of a performant node link cache.
		"""
		# Perform Initial Load
		## -> Presume update() is run before the first link is altered.
		## -> Else, the first link of the session will not update caches.
		## -> We still remain slightly unsure of the exact semantics.
		## -> Therefore, self.on_load() is also called as a load_post handler.
		if not hasattr(self, 'node_link_cache'):
			self.on_load()
			return

		# Register Validity Updater
		## -> They will be run after the update() method.
		## -> Between update() and set_validities, all is_valid=True are cleared.
		## -> Therefore, 'set_validities' only needs to set all is_valid=False.
		bpy.app.timers.register(self.node_link_cache.set_validities)

		# Ignore Updates
		## -> Certain corrective processes require suppressing the next update.
		## -> Otherwise, link corrections may trigger some nasty recursions.
		if not hasattr(self, 'ignore_update'):
			self.ignore_update = False

		# Regenerate NodeLinkCache
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
@bpy.app.handlers.persistent
def initialize_sim_tree_node_link_cache(_):
	"""Whenever a file is loaded, create/regenerate the NodeLinkCache in all trees."""
	for node_tree in bpy.data.node_groups:
		if node_tree.bl_idname == 'MaxwellSimTree':
			log.debug('%s: Initializing NodeLinkCache for NodeTree', str(node_tree))
			node_tree.on_load()


@bpy.app.handlers.persistent
def populate_missing_persistence(_) -> None:
	"""For all nodes and sockets with elements that don't have persistent elements computed, compute them.

	This is used when new dynamic enum properties are added to nodes and sockets, which need to first be computed and persisted in a context where setting properties is allowed.
	"""
	# Iterate over MaxwellSim Trees
	for node_tree in [
		_node_tree
		for _node_tree in bpy.data.node_groups
		if _node_tree.bl_idname == ct.TreeType.MaxwellSim.value and _node_tree.is_active
	]:
		log.debug(
			'%s: Regenerating Dynamic Field Persistance for NodeTree nodes/sockets',
			str(node_tree),
		)
		# Iterate over MaxwellSim Nodes
		# -> Excludes ex. frame and reroute nodes.
		for node in [_node for _node in node_tree.nodes if hasattr(_node, 'node_type')]:
			log.debug(
				'-> %s: Regenerating Dynamic Field Persistance for Node',
				str(node),
			)
			node.regenerate_dynamic_field_persistance()
			for bl_sockets in [node.inputs, node.outputs]:
				for bl_socket in bl_sockets:
					log.debug(
						'|-> %s: Regenerating Dynamic Field Persistance for Socket',
						str(bl_socket),
					)
					bl_socket.regenerate_dynamic_field_persistance()
	log.debug('Regenerated All Dynamic Field Persistance')


####################
# - Blender Registration
####################
bpy.app.handlers.load_post.append(initialize_sim_tree_node_link_cache)
# bpy.app.handlers.load_post.append(populate_missing_persistence)
## TODO: Move to top-level registration.

BL_REGISTER = [
	MaxwellSimTree,
]
