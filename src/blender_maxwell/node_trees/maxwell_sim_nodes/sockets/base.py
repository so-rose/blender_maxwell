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

import abc
import typing as typ

import bpy
import pydantic as pyd

from blender_maxwell.utils import bl_cache, bl_instance, logger, serialize

from .. import contracts as ct

log = logger.get(__name__)


####################
# - SocketDef
####################
class SocketDef(pyd.BaseModel, abc.ABC):
	"""Defines everything needed to initialize a `MaxwellSimSocket`.

	Used by nodes to specify which sockets to use as inputs/outputs.

	Notes:
		Not instantiated directly - rather, individual sockets each define a SocketDef subclass tailored to its own needs.

	Attributes:
		socket_type: The socket type to initialize.
	"""

	socket_type: ct.SocketType

	def preinit(self, bl_socket: bpy.types.NodeSocket) -> None:
		"""Pre-initialize a real Blender node socket from this socket definition.

		Parameters:
			bl_socket: The Blender node socket to alter using data from this SocketDef.
		"""
		bl_socket.reset_instance_id()
		bl_socket.regenerate_dynamic_field_persistance()

	def postinit(self, bl_socket: bpy.types.NodeSocket) -> None:
		"""Pre-initialize a real Blender node socket from this socket definition.

		Parameters:
			bl_socket: The Blender node socket to alter using data from this SocketDef.
		"""
		bl_socket.is_initializing = False
		bl_socket.on_active_kind_changed()

	@abc.abstractmethod
	def init(self, bl_socket: bpy.types.NodeSocket) -> None:
		"""Initializes a real Blender node socket from this socket definition.

		Parameters:
			bl_socket: The Blender node socket to alter using data from this SocketDef.
		"""

	####################
	# - Serialization
	####################
	def dump_as_msgspec(self) -> serialize.NaiveRepresentation:
		"""Transforms this `SocketDef` into an object that can be natively serialized by `msgspec`.

		Notes:
			Makes use of `pydantic.BaseModel.model_dump()` to cast any special fields into a serializable format.
			If this method is failing, check that `pydantic` can actually cast all the fields in your model.

		Returns:
			A particular `list`, with three elements:

			1. The `serialize`-provided "Type Identifier", to differentiate this list from generic list.
			2. The name of this subclass, so that the correct `SocketDef` can be reconstructed on deserialization.
			3. A dictionary containing simple Python types, as cast by `pydantic`.
		"""
		return [serialize.TypeID.SocketDef, self.__class__.__name__, self.model_dump()]

	@staticmethod
	def parse_as_msgspec(obj: serialize.NaiveRepresentation) -> typ.Self:
		"""Transforms an object made by `self.dump_as_msgspec()` into a subclass of `SocketDef`.

		Notes:
			The method presumes that the deserialized object produced by `msgspec` perfectly matches the object originally created by `self.dump_as_msgspec()`.

			This is a **mostly robust** presumption, as `pydantic` attempts to be quite consistent in how to interpret types with almost identical semantics.
			Still, yet-unknown edge cases may challenge these presumptions.

		Returns:
			A new subclass of `SocketDef`, initialized using the `model_dump()` dictionary.
		"""
		initialized_classes = [
			subclass(**obj[2])
			for subclass in SocketDef.__subclasses__()
			if subclass.__name__ == obj[1]
		]
		if not initialized_classes:
			msg = f'No "SocketDef" subclass found for name {obj[1]}. Please report this error'
			RuntimeError(msg)

		return initialized_classes[0]


####################
# - Socket
####################
MANDATORY_PROPS: set[str] = {'socket_type', 'bl_label'}


class MaxwellSimSocket(bpy.types.NodeSocket, bl_instance.BLInstance):
	"""A specialized Blender socket for nodes in a Maxwell simulation.

	Attributes:
		instance_id: A unique ID attached to a particular socket instance.
			Guaranteed to be unchanged so long as the socket lives.
			Used as a socket-specific cache index.
		locked: The lock-state of a particular socket, which determines the socket's user editability
	"""

	# Properties
	## Class
	socket_type: ct.SocketType
	bl_label: str

	## Computed by Subclass
	bl_idname: str

	# BLFields
	## Identifying
	is_initializing: bool = bl_cache.BLField(True, use_prop_update=False)

	active_kind: ct.FlowKind = bl_cache.BLField(ct.FlowKind.Value)

	## UI
	use_info_draw: bool = bl_cache.BLField(False, use_prop_update=False)
	use_prelock: bool = bl_cache.BLField(False, use_prop_update=False)

	locked: bool = bl_cache.BLField(False, use_prop_update=False)

	use_socket_color: bool = bl_cache.BLField(False, use_prop_update=False)
	socket_color: tuple[float, float, float, float] = bl_cache.BLField(
		(0, 0, 0, 0), use_prop_update=False
	)

	####################
	# - Initialization
	####################
	def __init_subclass__(cls, **kwargs: typ.Any):
		"""Initializes socket properties and attributes for use.

		Notes:
			Run when initializing any subclass of MaxwellSimSocket.
		"""
		log.debug('Initializing Socket: %s', cls.socket_type)
		super().__init_subclass__(**kwargs)
		cls.assert_attrs_valid(MANDATORY_PROPS)

		cls.bl_idname: str = str(cls.socket_type.value)

	####################
	# - Property Event: On Update
	####################
	def on_active_kind_changed(self) -> None:
		"""Matches the display shape to the active `FlowKind`.

		Notes:
			Called by `self.on_prop_changed()` when `self.active_kind` was changed.
		"""
		self.display_shape = self.active_kind.socket_shape

	def on_socket_prop_changed(self, prop_name: str) -> None:
		"""Called when a property has been updated.

		Notes:
			Can be overridden if a socket needs to respond to a property change.

			**Always prefer using node events instead of overriding this in a socket**.
			Think **very carefully** before using this, and use it with the greatest of care.

		Attributes:
			prop_name: The name of the property that was changed.
		"""

	def on_prop_changed(self, prop_name: str) -> None:
		"""Called when a property has been updated.

		Contrary to `node.on_prop_changed()`, socket-specific callbacks are baked into this function:

		- **Active Kind** (`self.active_kind`): Sets the socket shape to reflect the active `FlowKind`.
			**MAY NOT** rely on `FlowEvent` driven caches.
		- **Overrided Local Events** (`self.active_kind`): Sets the socket shape to reflect the active `FlowKind`.
			**MAY NOT** rely on `FlowEvent` driven caches.

		Attributes:
			prop_name: The name of the property that was changed.
		"""
		# All Attributes: Trigger Local Event
		## -> While initializing, only `DataChanged` won't trigger.
		if hasattr(self, prop_name):
			# Property Callbacks: Active Kind
			## -> WARNING: May NOT rely on flow.
			if prop_name == 'active_kind':
				self.on_active_kind_changed()

			# Property Callbacks: Per-Socket
			## -> WARNING: May NOT rely on flow.
			self.on_socket_prop_changed(prop_name)

			# Not Initializing: Trigger Event
			## -> This declares that the socket has changed.
			## -> This should happen first, in case dependents need a cache.
			if not self.is_initializing:
				self.trigger_event(ct.FlowEvent.DataChanged)

		# BLField Attributes: Invalidate BLField Dependents
		## -> Dependent props will generally also trigger on_prop_changed.
		## -> The recursion ends with the depschain.
		## -> WARNING: The chain is not checked for ex. cycles.
		if prop_name in self.blfields:
			self.invalidate_blfield_deps(prop_name)

	####################
	# - Link Event: Consent / On Change
	####################
	def allow_add_link(self, link: bpy.types.NodeLink) -> bool:
		"""Called to ask whether a link may be added to this (input) socket.

		- **Locked**: Locked sockets may not have links added.
		- **Capabilities**: Capabilities of both sockets participating in the link must be compatible.

		Notes:
			In practice, the link in question has already been added.
			This function determines **whether the new link should be instantly removed** - if so, the removal producing the _practical effect_ of the link "not being added" at all.

		Attributes:
			link: The node link that was already added, whose continued existance is in question.

		Returns:
			Whether or not consent is given to add the link.
			In practice, the link will simply remain if consent is given.
			If consent is not given, the new link will be removed.

		Raises:
			RuntimeError: If this socket is an output socket.
		"""
		# Output Socket Check
		if self.is_output:
			msg = f'Socket {self.bl_label} {self.socket_type}): Tried to ask output socket for consent to add link'
			raise RuntimeError(msg)

		# Lock Check
		if self.locked:
			log.error(
				'Attempted to link output socket "%s" (%s) to input socket "%s" (%s), but input socket is locked',
				link.from_socket.bl_label,
				link.from_socket.capabilities,
				self.bl_label,
				self.capabilities,
			)
			return False

		# Capability Check
		if not link.from_socket.capabilities.is_compatible_with(self.capabilities):
			log.error(
				'Attempted to link output socket "%s" (%s) to input socket "%s" (%s), but capabilities are incompatible',
				link.from_socket.bl_label,
				link.from_socket.capabilities,
				self.bl_label,
				self.capabilities,
			)
			return False

		return True

	def on_link_added(self, link: bpy.types.NodeLink) -> None:  # noqa: ARG002
		"""Triggers a `ct.FlowEvent.LinkChanged` event when a link is added.

		Notes:
			Called by the node tree, generally (but not guaranteed) after `self.allow_add_link()` has given consent to add the link.

		Attributes:
			link: The node link that was added.
				Currently unused.
		"""
		self.trigger_event(ct.FlowEvent.LinkChanged)

	def allow_remove_link(self, from_socket: bpy.types.NodeSocket) -> bool:  # noqa: ARG002
		"""Called to ask whether a link may be removed from this `to_socket`.

		- **Locked**: Locked sockets may not have links removed.
		- **Capabilities**: Capabilities of both sockets participating in the link must be compatible.

		Notes:
			In practice, the link in question has already been removed.
			Therefore, only the `from_socket` that the link _was_ attached to is provided.

		Attributes:
			from_socket: The node socket that was attached to before link removal.
				Currently unused.

		Returns:
			Whether or not consent is given to remove the link.
			If so, nothing will happen.
			If consent is not given, a new link will be added that is identical to the old one.

		Raises:
			RuntimeError: If this socket is an output socket.
		"""
		# Output Socket Check
		if self.is_output:
			msg = f'Socket {self.bl_label} {self.socket_type}): Tried to ask output socket for consent to remove link'
			raise RuntimeError(msg)

		# Lock Check
		if self.locked:
			return False

		return True

	def on_link_removed(self, from_socket: bpy.types.NodeSocket) -> None:  # noqa: ARG002
		"""Triggers a `ct.FlowEvent.LinkChanged` event when a link is removed.

		Notes:
			Called by the node tree, generally (but not guaranteed) after `self.allow_remove_link()` has given consent to remove the link.

		Attributes:
			from_socket: The node socket that was attached to before link removal.
				Currently unused.
		"""
		self.trigger_event(ct.FlowEvent.LinkChanged)

	def remove_invalidated_links(self) -> None:
		"""Reevaluates the capabilities of all socket links, and removes any that no longer match.

		Links are removed with a simple `node_tree.links.remove()`, which directly emulates a user trying to remove the node link.
		**Note** that all of the usual consent-semantics apply just the same as if the user had manually tried to remove the link.

		Notes:
			Called by nodes directly on their sockets, after altering any property that might influence the capabilities of that socket.

			This prevents invalid use when the user alters a property, which **would** disallow adding a _new_ link identical to one that already exists.
			In such a case, the existing (non-capability-respecting) link should be removed, as it has become invalid.
		"""
		node_tree = self.id_data
		for link in self.links:
			if not link.from_socket.capabilities.is_compatible_with(
				link.to_socket.capabilities
			):
				log.error(
					'Deleted link between "%s" (%s) and "%s" (%s) due to invalidated capabilities',
					link.from_socket.bl_label,
					link.from_socket.capabilities,
					link.to_socket.bl_label,
					link.to_socket.capabilities,
				)
				node_tree.links.remove(link)

	####################
	# - Event Chain
	####################
	def trigger_event(
		self,
		event: ct.FlowEvent,
		socket_kinds: set[ct.FlowKind] | None = None,
	) -> None:
		"""Responds to and triggers subsequent events along the node tree.

		- **Locking**: `EnableLock` or `DisableLock` will always affect this socket's lock.
		- **Input Socket -> Input**: Trigger event on `from_socket`s along input links.
		- **Input Socket -> Output**: Trigger event on node (w/`socket_name`).
		- **Output Socket -> Input**: Trigger event on node (w/`socket_name`).
		- **Output Socket -> Output**: Trigger event on `to_socket`s along output links.


		Notes:
			This can be an unpredictably heavy function, depending on the node graph topology.

			A `LinkChanged` (->Output) event will trigger a `DataChanged` event on the node.
			**This may change** if it becomes important for the node to differentiate between "change in data" and "change in link".

		Parameters:
			event: The event to report along the node tree.
				The value of `ct.FlowEvent.flow_direction[event]` (`input` or `output`) determines the direction that an event flows.
		"""
		flow_direction = ct.FlowEvent.flow_direction[event]

		# Locking
		if event in [ct.FlowEvent.EnableLock, ct.FlowEvent.DisableLock]:
			self.locked = event == ct.FlowEvent.EnableLock

		# Event by Socket Orientation | Flow Direction
		match (self.is_output, flow_direction):
			case (False, 'input'):
				for link in self.links:
					link.from_socket.trigger_event(event, socket_kinds=socket_kinds)

			case (False, 'output'):
				if event == ct.FlowEvent.LinkChanged:
					self.node.trigger_event(
						ct.FlowEvent.DataChanged,
						socket_name=self.name,
						socket_kinds=socket_kinds,
					)
				else:
					self.node.trigger_event(
						event, socket_name=self.name, socket_kinds=socket_kinds
					)

			case (True, 'input'):
				self.node.trigger_event(
					event, socket_name=self.name, socket_kinds=socket_kinds
				)

			case (True, 'output'):
				for link in self.links:
					link.to_socket.trigger_event(event, socket_kinds=socket_kinds)

	####################
	# - FlowKind: Auxiliary
	####################
	# Capabilities
	@property
	def capabilities(self) -> None:
		"""By default, the socket is linkeable with any other socket of the same type and active kind.

		Notes:
			See `ct.FlowKind` for more information.
		"""
		return ct.CapabilitiesFlow(
			socket_type=self.socket_type,
			active_kind=self.active_kind,
		)

	# Info
	@property
	def info(self) -> ct.InfoFlow:
		"""Signal that no information is declared by this socket.

		Notes:
			See `ct.FlowKind` for more information.

		Returns:
			An empty `ct.InfoFlow`.
		"""
		return ct.FlowSignal.NoFlow

	# Param
	@property
	def params(self) -> ct.ParamsFlow:
		"""Signal that no params are declared by this socket.

		Notes:
			See `ct.FlowKind` for more information.

		Returns:
			An empty `ct.ParamsFlow`.
		"""
		return ct.FlowSignal.NoFlow

	####################
	# - FlowKind: Auxiliary
	####################
	# Value
	@property
	def value(self) -> ct.ValueFlow:
		"""Throws a descriptive error.

		Notes:
			See `ct.FlowKind` for more information.

		Raises:
			NotImplementedError: When used without being overridden.
		"""
		return ct.FlowSignal.NoFlow

	@value.setter
	def value(self, value: ct.ValueFlow) -> None:
		"""Throws a descriptive error.

		Notes:
			See `ct.FlowKind` for more information.

		Raises:
			NotImplementedError: When used without being overridden.
		"""
		msg = f'Socket {self.bl_label} {self.socket_type}): Tried to set "ct.FlowKind.Value", but socket does not define it'
		raise NotImplementedError(msg)

	# Array
	@property
	def array(self) -> ct.ArrayFlow:
		"""Throws a descriptive error.

		Notes:
			See `ct.FlowKind` for more information.

		Raises:
			NotImplementedError: When used without being overridden.
		"""
		return ct.FlowSignal.NoFlow

	@array.setter
	def array(self, value: ct.ArrayFlow) -> None:
		"""Throws a descriptive error.

		Notes:
			See `ct.FlowKind` for more information.

		Raises:
			NotImplementedError: When used without being overridden.
		"""
		msg = f'Socket {self.bl_label} {self.socket_type}): Tried to set "ct.FlowKind.Array", but socket does not define it'
		raise NotImplementedError(msg)

	# Func
	@property
	def lazy_func(self) -> ct.FuncFlow:
		"""Throws a descriptive error.

		Notes:
			See `ct.FlowKind` for more information.

		Raises:
			NotImplementedError: When used without being overridden.
		"""
		return ct.FlowSignal.NoFlow

	@lazy_func.setter
	def lazy_func(self, lazy_func: ct.FuncFlow) -> None:
		"""Throws a descriptive error.

		Notes:
			See `ct.FlowKind` for more information.

		Raises:
			NotImplementedError: When used without being overridden.
		"""
		msg = f'Socket {self.bl_label} {self.socket_type}): Tried to set "ct.FlowKind.Func", but socket does not define it'
		raise NotImplementedError(msg)

	# Range
	@property
	def lazy_range(self) -> ct.RangeFlow:
		"""Throws a descriptive error.

		Notes:
			See `ct.FlowKind` for more information.

		Raises:
			NotImplementedError: When used without being overridden.
		"""
		return ct.FlowSignal.NoFlow

	@lazy_range.setter
	def lazy_range(self, value: ct.RangeFlow) -> None:
		"""Throws a descriptive error.

		Notes:
			See `ct.FlowKind` for more information.

		Raises:
			NotImplementedError: When used without being overridden.
		"""
		msg = f'Socket {self.bl_label} {self.socket_type}): Tried to set "ct.FlowKind.Range", but socket does not define it'
		raise NotImplementedError(msg)

	####################
	# - Data Chain Computation
	####################
	def _compute_data(
		self,
		kind: ct.FlowKind = ct.FlowKind.Value,
	) -> typ.Any:
		"""Low-level method to computes the data contained within this socket, for a particular `ct.FlowKind`.

		Notes:
			Not all `ct.FlowKind`s are meant to be computed; namely, `Capabilities` should be directly referenced.

		Raises:
			ValueError: When referencing a socket that's meant to be directly referenced.
		"""
		kind_data_map = {
			ct.FlowKind.Value: lambda: self.value,
			ct.FlowKind.Array: lambda: self.array,
			ct.FlowKind.Func: lambda: self.lazy_func,
			ct.FlowKind.Range: lambda: self.lazy_range,
			ct.FlowKind.Params: lambda: self.params,
			ct.FlowKind.Info: lambda: self.info,
		}
		if kind in kind_data_map:
			return kind_data_map[kind]()

		## TODO: Reflect this constraint in the type
		msg = f'Socket {self.bl_label} ({self.socket_type}): Kind {kind} cannot be computed within a socket "compute_data", as it is meant to be referenced directly'
		raise ValueError(msg)

	def compute_data(
		self,
		kind: ct.FlowKind = ct.FlowKind.Value,
	) -> typ.Any:
		"""Computes internal or link-sourced data represented by this socket.

		- **Input Socket | Unlinked**: Use socket's own data, by calling `_compute_data`.
		- **Input Socket | Linked**: Call `compute_data` on the linked `from_socket`.
		- **Output Socket**: Use the node's output data, by calling `node.compute_output()`.

		Notes:
			This can be an unpredictably heavy function, depending on the node graph topology.

		Parameters:
			kind: The `ct.FlowKind` to reference when retrieving the data.

		Returns:
			The computed data, whever it came from.

		Raises:
			NotImplementedError: If multi-input sockets are used (no support yet as of Blender 4.1).
		"""
		# Compute Output Socket
		if self.is_output:
			return self.node.compute_output(self.name, kind=kind)

		# Compute Input Socket
		## Unlinked: Retrieve Socket Value
		if not self.is_linked:
			return self._compute_data(kind)

		## Linked: Compute Data on Linked Socket
		## -> Capabilities are guaranteed compatible by 'allow_link_add'.
		## -> There is no point in rechecking every time data flows.
		linked_values = [link.from_socket.compute_data(kind) for link in self.links]

		# Return Single Value / List of Values
		if len(linked_values) == 1:
			return linked_values[0]

		# Edge Case: While Dragging Link (but not yet removed)
		## While the user is dragging a link:
		## - self.is_linked = True, since the user hasn't confirmed anything.
		## - self.links will be empty, since the link object was freed.
		## When this particular condition is met, pretend that we're not linked.
		if len(linked_values) == 0:
			return self._compute_data(kind)

		msg = f'Socket {self.bl_label} ({self.socket_type}): Multi-input sockets are not yet supported'
		raise NotImplementedError(msg)

	####################
	# - UI - Color
	####################
	def draw_color(
		self,
		_: bpy.types.Context,
		node: bpy.types.Node,  # noqa: ARG002
	) -> tuple[float, float, float, float]:
		"""Draw the socket color depending on context.

		When `self.use_socket_color` is set, the property `socket_color` can be used to control the socket color directly.
		Otherwise, a default based on `self.socket_type` will be used.

		Notes:
			Called by Blender to call the socket color.
		"""
		if self.use_socket_color:
			return self.socket_color
		return ct.SOCKET_COLORS[self.socket_type]

	####################
	# - UI
	####################
	def draw(
		self,
		context: bpy.types.Context,
		layout: bpy.types.UILayout,
		node: bpy.types.Node,
		text: str,
	) -> None:
		"""Draw the socket UI.

		- **Input Socket**: Will use `self.draw_input()`.
		- **Output Socket**: Will use `self.draw_output()`.

		Parameters:
			context: The current Blender context.
			layout: Target for defining UI elements.
			node: The node within which the socket is embedded.
			text: The socket's name in the UI.
		"""
		if self.is_output:
			self.draw_output(context, layout, node, text)
		else:
			self.draw_input(context, layout, node, text)

	def draw_prelock(
		self,
		context: bpy.types.Context,
		col: bpy.types.UILayout,
		node: bpy.types.Node,
		text: str,
	) -> None:
		"""Draw the "prelock" UI, which is usable regardless of the `self.locked` state.

		Notes:
			If a "prelock" UI is needed by a socket, it should set `self.use_prelock` and override this method.

		Parameters:
			context: The current Blender context.
			col: Target for defining UI elements.
			node: The node within which the socket is embedded.
			text: The socket's name in the UI.
		"""

	####################
	# - UI: Input / Output Socket
	####################
	def draw_input(
		self,
		context: bpy.types.Context,
		layout: bpy.types.UILayout,
		node: bpy.types.Node,
		text: str,
	) -> None:
		"""Draw the UI of the input socket.

		- **Locked** (`self.locked`): The UI will be unusable.
		- **Linked** (`self.is_linked`): Only the socket label will display.
		- **Use Prelock** (`self.use_prelock`): The "prelock" UI drawn with `self.draw_prelock()`, which shows **regardless of `self.locked`**.
		- **FlowKind**: The `FlowKind`-specific UI corresponding to the current `self.active_kind`.

		Notes:
			Shouldn't be overridden.

		Parameters:
			context: The current Blender context.
			layout: Target for defining UI elements.
			node: The node within which the socket is embedded.
			text: The socket's name in the UI.
		"""
		col = layout.column()

		# Row: Label
		row = col.row()
		row.alignment = 'LEFT'

		## Lock Check
		if self.locked:
			row.enabled = False

		## Link Check
		if self.is_linked:
			self.draw_input_label_row(row, text)
		else:
			self.draw_label_row(row, text)

			# User Prelock Row
			row = col.row(align=False)
			if self.use_prelock:
				_col = row.column(align=False)
				_col.enabled = True
				self.draw_prelock(context, _col, node, text)

				if self.locked:
					row = col.row(align=False)
					row.enabled = False
			elif self.locked:
				row.enabled = False

			# FlowKind Draw Row
			col = row.column(align=True)
			{
				ct.FlowKind.Value: self.draw_value,
				ct.FlowKind.Array: self.draw_array,
				ct.FlowKind.Range: self.draw_lazy_range,
				ct.FlowKind.Func: self.draw_lazy_func,
			}[self.active_kind](col)

		# Info Drawing
		if self.use_info_draw:
			info = self.compute_data(kind=ct.FlowKind.Info)
			if not ct.FlowSignal.check(info):
				self.draw_info(info, col)

	def draw_output(
		self,
		context: bpy.types.Context,  # noqa: ARG002
		layout: bpy.types.UILayout,
		node: bpy.types.Node,  # noqa: ARG002
		text: str,
	) -> None:
		"""Draw the label text on the output socket.

		Notes:
			Shouldn't be overridden.

		Parameters:
			context: The current Blender context.
			layout: Target for defining UI elements.
			node: The node within which the socket is embedded.
			text: The socket's name in the UI.
		"""
		col = layout.column()

		# Row: Label
		row = col.row()
		row.alignment = 'RIGHT'
		self.draw_output_label_row(row, text)

		# Draw FlowKind.Info related Information
		if self.use_info_draw:
			info = self.compute_data(kind=ct.FlowKind.Info)
			if not ct.FlowSignal.check(info):
				self.draw_info(info, col)

	####################
	# - UI Methods: Label Rows
	####################
	def draw_label_row(
		self,
		row: bpy.types.UILayout,
		text: str,
	) -> None:
		"""Draw the label row, which is at the same height as the socket shape.

		Will only display if the socket is an **unlinked input socket**.

		Notes:
			Can be overriden by individual socket classes, if they need to alter the way that the label row is drawn.

		Parameters:
			row: Target for defining UI elements.
			text: The socket's name in the UI.
		"""
		row.label(text=text)

	def draw_input_label_row(
		self,
		row: bpy.types.UILayout,
		text: str,
	) -> None:
		"""Draw the label row, which is at the same height as the socket shape.

		Will only display if the socket is a **linked input socket**.

		Notes:
			Can be overriden by individual socket classes, if they need to alter the way that the label row is drawn.

		Parameters:
			row: Target for defining UI elements.
			text: The socket's name in the UI.
		"""
		row.label(text=text)

	def draw_output_label_row(
		self,
		row: bpy.types.UILayout,
		text: str,
	) -> None:
		"""Draw the output label row, which is at the same height as the socket shape.

		Will only display if the socket is an **output socket**.

		Notes:
			Can be overriden by individual socket classes, if they need to alter the way that the output label row is drawn.

		Parameters:
			row: Target for defining UI elements.
			text: The socket's name in the UI.
		"""
		row.label(text=text)

	####################
	# - UI Methods: Active FlowKind
	####################
	def draw_value(self, col: bpy.types.UILayout) -> None:
		"""Draws the socket value on its own line.

		Notes:
			Should be overriden by individual socket classes, if they have an editable `FlowKind.Value`.

		Parameters:
			col: Target for defining UI elements.
		"""

	def draw_lazy_range(self, col: bpy.types.UILayout) -> None:
		"""Draws the socket lazy array range on its own line.

		Notes:
			Should be overriden by individual socket classes, if they have an editable `FlowKind.Range`.

		Parameters:
			col: Target for defining UI elements.
		"""

	def draw_array(self, col: bpy.types.UILayout) -> None:
		"""Draws the socket array UI on its own line.

		Notes:
			Should be overriden by individual socket classes, if they have an editable `FlowKind.Array`.

		Parameters:
			col: Target for defining UI elements.
		"""

	def draw_lazy_func(self, col: bpy.types.UILayout) -> None:
		"""Draws the socket lazy value function UI on its own line.

		Notes:
			Should be overriden by individual socket classes, if they have an editable `FlowKind.Func`.

		Parameters:
			col: Target for defining UI elements.
		"""

	####################
	# - UI Methods: Auxilliary
	####################
	def draw_info(self, info: ct.InfoFlow, col: bpy.types.UILayout) -> None:
		"""Draws the socket info on its own line.

		Notes:
			Should be overriden by individual socket classes, if they might output a `FlowKind.Info`.

		Parameters:
			col: Target for defining UI elements.
		"""
