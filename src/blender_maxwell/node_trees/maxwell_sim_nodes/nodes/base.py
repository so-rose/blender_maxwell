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

"""Defines a special base class, `MaxwellSimNode`, from which all nodes inherit.

Attributes:
	MANDATORY_PROPS: Properties that must be defined on the `MaxwellSimNode`.
"""

import enum
import typing as typ
from collections import defaultdict
from types import MappingProxyType

import bpy
import sympy as sp

from blender_maxwell.utils import bl_cache, bl_instance, logger

from .. import contracts as ct
from .. import managed_objs as _managed_objs
from .. import sockets
from . import events
from . import presets as _presets

log = logger.get(__name__)

####################
# - Types
####################
Sockets: typ.TypeAlias = dict[str, sockets.base.SocketDef]
Preset: typ.TypeAlias = dict[str, _presets.PresetDef]
ManagedObjs: typ.TypeAlias = dict[ct.ManagedObjName, type[_managed_objs.ManagedObj]]

MANDATORY_PROPS: set[str] = {'node_type', 'bl_label'}


####################
# - Node
####################
class MaxwellSimNode(bpy.types.Node, bl_instance.BLInstance):
	"""A specialized Blender node for Maxwell simulations.

	Attributes:
		node_type: The `ct.NodeType` that identifies which node this is.
		bl_label: The label shown in the header of the node in Blender.
		instance_id: A unique ID attached to a particular node instance.
			Guaranteed to be unchanged so long as the node lives.
			Used as a node-specific cache index.
		sim_node_name: A unique human-readable name identifying the node.
			Used when naming managed objects and exporting.
		preview_active: Whether the preview (if any) is currently active.
		locked: Whether the node is currently 'locked' aka. non-editable.
	"""

	####################
	# - Properties
	####################
	node_type: ct.NodeType
	bl_label: str

	# Features
	use_sim_node_name: bool = False

	# Declarations
	input_sockets: typ.ClassVar[Sockets] = MappingProxyType({})
	output_sockets: typ.ClassVar[Sockets] = MappingProxyType({})

	input_socket_sets: typ.ClassVar[dict[str, Sockets]] = MappingProxyType({})
	output_socket_sets: typ.ClassVar[dict[str, Sockets]] = MappingProxyType({})

	managed_obj_types: typ.ClassVar[ManagedObjs] = MappingProxyType({})
	presets: typ.ClassVar[dict[str, Preset]] = MappingProxyType({})

	## __init_subclass__ Computed
	bl_idname: str

	####################
	# - Fields
	####################
	sim_node_name: str = bl_cache.BLField('')

	# Loose Sockets
	loose_input_sockets: dict[str, sockets.base.SocketDef] = bl_cache.BLField({})
	loose_output_sockets: dict[str, sockets.base.SocketDef] = bl_cache.BLField({})

	# UI Options
	preview_active: bool = bl_cache.BLField(False)
	locked: bool = bl_cache.BLField(False, use_prop_update=False)

	# Active Socket Set
	active_socket_set: enum.StrEnum = bl_cache.BLField(
		enum_cb=lambda self, _: self.socket_sets_bl_enum()
	)

	@classmethod
	def socket_sets_bl_enum(cls) -> list[ct.BLEnumElement]:
		return [
			(socket_set_name, socket_set_name, socket_set_name, '', i)
			for i, socket_set_name in enumerate(cls.socket_set_names())
		]

	# Active Preset
	active_preset: enum.StrEnum = bl_cache.BLField(
		enum_cb=lambda self, _: self.presets_bl_enum()
	)

	@classmethod
	def presets_bl_enum(cls) -> list[ct.BLEnumElement]:
		return [
			(
				preset_name,
				preset_def.label,
				preset_def.description,
				'',
				i,
			)
			for i, (preset_name, preset_def) in enumerate(cls.presets.items())
		]

	# Managed Objects
	managed_objs: dict[str, _managed_objs.ManagedObj] = bl_cache.BLField(
		{}, use_prop_update=False
	)

	####################
	# - Class Methods
	####################
	@classmethod
	def socket_set_names(cls) -> list[str]:
		"""Retrieve the names of socket sets, in an order-preserving way.

		Notes:
			Semantically similar to `list(set(...) | set(...))`.

		Returns:
			List of socket set names, without duplicates, in definition order.
		"""
		return (_input_socket_set_names := list(cls.input_socket_sets.keys())) + [
			output_socket_set_name
			for output_socket_set_name in cls.output_socket_sets
			if output_socket_set_name not in _input_socket_set_names
		]

	@classmethod
	def _gather_event_methods(cls) -> dict[str, typ.Callable[[], None]]:
		"""Gathers all methods called in response to events observed by the node.

		Notes:
			- 'Event methods' must have an attribute 'event' in order to be picked up.
			- 'Event methods' must have an attribute 'event'.

		Returns:
			Event methods, indexed by the event that (maybe) triggers them.
		"""
		event_methods = [
			method
			for attr_name in dir(cls)
			if hasattr(method := getattr(cls, attr_name), 'event')
			and method.event in set(ct.FlowEvent)
			## Forbidding blfields prevents triggering __get__ on bl_property
		]
		event_methods_by_event = {event: [] for event in set(ct.FlowEvent)}
		for method in event_methods:
			event_methods_by_event[method.event].append(method)

		return event_methods_by_event

	####################
	# - Subclass Initialization
	####################
	@classmethod
	def __init_subclass__(cls, **kwargs) -> None:
		"""Initializes node properties and attributes for use.

		Notes:
			Run when initializing any subclass of MaxwellSimNode.
		"""
		log.debug('Initializing Node: %s', cls.node_type)
		super().__init_subclass__(**kwargs)

		# Check Attribute Validity
		cls.assert_attrs_valid(MANDATORY_PROPS)

		# Node Properties
		cls.bl_idname: str = str(cls.node_type.value)
		cls.event_methods_by_event = cls._gather_event_methods()

	####################
	# - Events: Sim Node Name | Active Socket Set | Active Preset
	####################
	@events.on_value_changed(
		prop_name='sim_node_name',
		props={'sim_node_name', 'managed_objs', 'managed_obj_types'},
		stop_propagation=True,
	)
	def _on_sim_node_name_changed(self, props):
		log.debug(
			'Changed Sim Node Name of a "%s" to "%s" (self=%s)',
			self.bl_idname,
			props['sim_node_name'],
			str(self),
		)

		# (Re)Construct Managed Objects
		## -> Due to 'prev_name', the new MObjs will be renamed on construction
		self.managed_objs = {
			mobj_name: mobj_type(
				self.sim_node_name,
				prev_name=(
					props['managed_objs'][mobj_name].name
					if mobj_name in props['managed_objs']
					else None
				),
			)
			for mobj_name, mobj_type in props['managed_obj_types'].items()
		}

	@events.on_value_changed(prop_name='active_socket_set')
	def _on_socket_set_changed(self):
		log.info(
			'Changed Sim Node Socket Set to "%s"',
			self.active_socket_set,
		)
		self._sync_sockets()

	@events.on_value_changed(
		prop_name='active_preset',
		run_on_init=True,
		props={'presets', 'active_preset'},
		stop_propagation=True,
	)
	def _on_active_preset_changed(self, props: dict):
		if props['active_preset'] is not None:
			log.info(
				'Changed Sim Node Preset to "%s"',
				props['active_preset'],
			)

			# Retrieve Preset
			if not (preset_def := props['presets'].get(props['active_preset'])):
				msg = f'Tried to apply active preset, but the active preset "{props["active_preset"]}" is not a defined preset: {props["active_preset"]}'
				raise RuntimeError(msg)

			# Apply Preset to Sockets
			for socket_name, socket_value in preset_def.values.items():
				if not (bl_socket := self.inputs.get(socket_name)):
					msg = f'Tried to set preset socket/value pair ({socket_name}={socket_value}), but socket is not in active input sockets ({self.inputs})'
					raise ValueError(msg)

				## TODO: Account for FlowKind
				bl_socket.value = socket_value

	####################
	# - Events: Preview | Plot
	####################
	@events.on_show_plot(stop_propagation=False)
	def _on_show_plot(self):
		node_tree = self.id_data
		if len(self.event_methods_by_event[ct.FlowEvent.ShowPlot]) > 1:
			## TODO: Is this check good enough?
			## TODO: Set icon/indicator/something to make it clear which node is being previewed.
			node_tree.report_show_plot(self)

	@events.on_show_preview()
	def _on_show_preview(self):
		node_tree = self.id_data
		node_tree.report_show_preview(self)

		# Set Preview to Active
		## Implicitly triggers any @on_value_changed for preview_active.
		if not self.preview_active:
			self.preview_active = True

	@events.on_value_changed(
		prop_name='preview_active', props={'preview_active'}, stop_propagation=True
	)
	def _on_preview_changed(self, props):
		if not props['preview_active']:
			for mobj in self.managed_objs.values():
				mobj.hide_preview()

	####################
	# - Events: Lock
	####################
	@events.on_enable_lock()
	def _on_enabled_lock(self):
		# Set Locked to Active
		## draw() picks up on this immediately.
		## Doesn't trigger @on_value_changed, since self.locked has no update().
		self.locked = True

	@events.on_disable_lock()
	def _on_disabled_lock(self):
		# Set Locked to Inactive
		## draw() picks up on this immediately.
		## Doesn't trigger @on_value_changed, since self.locked has no update().
		self.locked = False

	####################
	# - Events: Loose Sockets
	####################
	@events.on_value_changed(prop_name={'loose_input_sockets', 'loose_output_sockets'})
	def _on_loose_sockets_changed(self):
		self._sync_sockets()

	####################
	# - Socket Accessors
	####################
	def _bl_sockets(
		self, direc: typ.Literal['input', 'output']
	) -> bpy.types.NodeInputs:
		"""Retrieve currently visible Blender sockets on the node, by-direction.

		Only use internally, when `node.inputs`/`node.outputs` is too much of a mouthful to use directly.

		Notes:
			You should probably use `node.inputs` or `node.outputs` directly.

		Parameters:
			direc: The direction to load Blender sockets from.

		Returns:
			The actual `node.inputs` or `node.outputs`, depending on `direc`.
		"""
		return self.inputs if direc == 'input' else self.outputs

	def _active_socket_set_socket_defs(
		self,
		direc: typ.Literal['input', 'output'],
	) -> dict[ct.SocketName, sockets.base.SocketDef]:
		"""Retrieve all socket definitions for sockets that should be defined, according to the `self.active_socket_set`.

		Notes:
			You should probably use `self.active_socket_defs()`

		Parameters:
			direc: The direction to load Blender sockets from.

		Returns:
			Mapping from socket names to corresponding `sockets.base.SocketDef`s.

			If `self.active_socket_set` is None, the empty dict is returned.
		"""
		# No Active Socket Set: Return Nothing
		if self.active_socket_set is None:
			return {}

		# Retrieve Active Socket Set Sockets
		socket_sets = (
			self.input_socket_sets if direc == 'input' else self.output_socket_sets
		)
		return socket_sets.get(self.active_socket_set, {})

	def active_socket_defs(
		self, direc: typ.Literal['input', 'output']
	) -> dict[ct.SocketName, sockets.base.SocketDef]:
		"""Retrieve all socket definitions for sockets that should be defined.

		Parameters:
			direc: The direction to load Blender sockets from.

		Returns:
			Mapping from socket names to corresponding `sockets.base.SocketDef`s.
		"""
		static_sockets = self.input_sockets if direc == 'input' else self.output_sockets
		loose_sockets = (
			self.loose_input_sockets if direc == 'input' else self.loose_output_sockets
		)

		return (
			static_sockets
			| self._active_socket_set_socket_defs(direc=direc)
			| loose_sockets
		)

	####################
	# - Socket Management
	####################
	## TODO: Check for namespace collisions in sockets to prevent silent errors
	def _prune_inactive_sockets(self):
		"""Remove all "inactive" sockets from the node.

		A socket is considered "inactive" when it shouldn't be defined (per `self.active_socket_defs), but is present nonetheless.
		"""
		node_tree = self.id_data
		for direc in ['input', 'output']:
			all_bl_sockets = self._bl_sockets(direc)
			active_bl_socket_defs = self.active_socket_defs(direc)

			# Determine Sockets to Remove
			bl_sockets_to_remove = [
				bl_socket
				for socket_name, bl_socket in all_bl_sockets.items()
				if socket_name not in active_bl_socket_defs
				or socket_name
				in (
					self.loose_input_sockets
					if direc == 'input'
					else self.loose_output_sockets
				)
			]

			# Remove Sockets
			for bl_socket in bl_sockets_to_remove:
				bl_socket_name = bl_socket.name

				# 1. Report the socket removal to the NodeTree.
				## -> The NodeLinkCache needs to be adjusted manually.
				node_tree.on_node_socket_removed(bl_socket)

				# 2. Invalidate the input socket cache across all kinds.
				## -> Prevents phantom values from remaining available.
				self._compute_input.invalidate(
					input_socket_name=bl_socket_name,
					kind=...,
					unit_system=...,
				)

				# 3. Perform the removal using Blender's API.
				## -> Actually removes the socket.
				all_bl_sockets.remove(bl_socket)

				if direc == 'input':
					# 4. Run all trigger-only `on_value_changed` callbacks.
					## -> Runs any event methods that relied on the socket.
					## -> Only methods that don't **require** the socket.
					## Trigger-Only: If method loads no socket data, it runs.
					## `optional`: If method optional-loads socket, it runs.
					triggered_event_methods = [
						event_method
						for event_method in self.filtered_event_methods_by_event(
							ct.FlowEvent.DataChanged, (bl_socket_name, None, None)
						)
						if bl_socket_name
						not in event_method.callback_info.must_load_sockets
					]
					for event_method in triggered_event_methods:
						log.critical(
							'%s: Running %s',
							self.sim_node_name,
							str(event_method),
						)
						event_method(self)

	def _add_new_active_sockets(self):
		"""Add and initialize all "active" sockets that aren't on the node.

		Existing sockets within the given direction are not re-created.
		"""
		for direc in ['input', 'output']:
			all_bl_sockets = self._bl_sockets(direc)
			active_bl_socket_defs = self.active_socket_defs(direc)

			# Define BL Sockets
			created_sockets = {}
			for socket_name, socket_def in active_bl_socket_defs.items():
				# Skip Existing Sockets
				if socket_name in all_bl_sockets:
					continue

				# Create BL Socket from Socket
				## Set 'display_shape' from 'socket_shape'
				all_bl_sockets.new(
					str(socket_def.socket_type.value),
					socket_name,
				)

				# Record Socket Creation
				created_sockets[socket_name] = socket_def

			# Initialize Just-Created BL Sockets
			for socket_name, socket_def in created_sockets.items():
				socket_def.preinit(all_bl_sockets[socket_name])
				socket_def.init(all_bl_sockets[socket_name])
				socket_def.postinit(all_bl_sockets[socket_name])

	def _sync_sockets(self) -> None:
		"""Synchronize the node's sockets with the active sockets.

		- Any non-existing active socket will be added and initialized.
		- Any existing active socket will not be changed.
		- Any existing inactive socket will be removed.

		Notes:
			Must be called after any change to socket definitions, including loose
		sockets.
		"""
		self._prune_inactive_sockets()
		self._add_new_active_sockets()

	####################
	# - Event Methods
	####################
	@property
	def _event_method_filter_by_event(self) -> dict[ct.FlowEvent, typ.Callable]:
		"""Compute a map of FlowEvents, to a function that filters its event methods.

		The returned filter functions are hard-coded, and must always return a `bool`.
		They may use attributes of `self`, always return `True` or `False`, or something different.

		Notes:
			This is an internal method; you probably want `self.filtered_event_methods_by_event`.

		Returns:
			The map of `ct.FlowEvent` to a function that can determine whether any `event_method` should be run.
		"""
		return {
			ct.FlowEvent.EnableLock: lambda *_: True,
			ct.FlowEvent.DisableLock: lambda *_: True,
			ct.FlowEvent.DataChanged: lambda event_method, socket_name, prop_name, _: (
				(
					socket_name
					and socket_name in event_method.callback_info.on_changed_sockets
				)
				or (
					prop_name
					and prop_name in event_method.callback_info.on_changed_props
				)
				or (
					socket_name
					and event_method.callback_info.on_any_changed_loose_input
					and socket_name in self.loose_input_sockets
				)
			),
			ct.FlowEvent.OutputRequested: lambda output_socket_method,
			output_socket_name,
			_,
			kind: (
				kind == output_socket_method.callback_info.kind
				and (
					output_socket_name
					== output_socket_method.callback_info.output_socket_name
				)
			),
			ct.FlowEvent.ShowPreview: lambda *_: True,
			ct.FlowEvent.ShowPlot: lambda *_: True,
		}

	def filtered_event_methods_by_event(
		self,
		event: ct.FlowEvent,
		_filter: tuple[ct.SocketName, str],
	) -> list[typ.Callable]:
		"""Return all event methods that should run, given the context provided by `_filter`.

		The inclusion decision is made by the internal property `self._event_method_filter_by_event`.

		Returns:
			All `event_method`s that should run, as callable objects (they can be run using `event_method(self)`).
		"""
		return [
			event_method
			for event_method in self.event_methods_by_event[event]
			if self._event_method_filter_by_event[event](event_method, *_filter)
		]

	####################
	# - Compute: Input Socket
	####################
	@bl_cache.keyed_cache(
		exclude={'self', 'optional'},
		encode={'unit_system'},
	)
	def _compute_input(
		self,
		input_socket_name: ct.SocketName,
		kind: ct.FlowKind = ct.FlowKind.Value,
		unit_system: dict[ct.SocketType, sp.Expr] | None = None,
		optional: bool = False,
	) -> typ.Any:
		"""Computes the data of an input socket, following links if needed.

		Notes:
			The semantics derive entirely from `sockets.MaxwellSimSocket.compute_data()`.

		Parameters:
			input_socket_name: The name of the input socket to compute the value of.
				It must be currently active.
			kind: The data flow kind to compute.
		"""
		bl_socket = self.inputs.get(input_socket_name)
		if bl_socket is not None:
			if bl_socket.instance_id:
				return (
					ct.FlowKind.scale_to_unit_system(
						kind,
						bl_socket.compute_data(kind=kind),
						unit_system,
					)
					if unit_system is not None
					else bl_socket.compute_data(kind=kind)
				)

			# No Socket Instance ID
			## -> Indicates that socket_def.preinit() has not yet run.
			## -> Anyone needing results will need to wait on preinit().
			return ct.FlowSignal.FlowInitializing

		if optional:
			return ct.FlowSignal.NoFlow

		msg = f'{self.sim_node_name}: Input socket "{input_socket_name}" cannot be computed, as it is not an active input socket'
		raise ValueError(msg)

	####################
	# - Compute Event: Output Socket
	####################
	@bl_cache.keyed_cache(
		exclude={'self', 'optional'},
	)
	def compute_output(
		self,
		output_socket_name: ct.SocketName,
		kind: ct.FlowKind = ct.FlowKind.Value,
		optional: bool = False,
	) -> typ.Any:
		"""Computes the value of an output socket.

		Parameters:
			output_socket_name: The name declaring the output socket, for which this method computes the output.
			kind: The FlowKind to use when computing the output socket value.

		Returns:
			The value of the output socket, as computed by the dedicated method
			registered using the `@computes_output_socket` decorator.
		"""
		if self.outputs.get(output_socket_name) is None:
			if optional:
				return None

			msg = f"Can't compute nonexistent output socket name {output_socket_name}, as it's not currently active"
			raise RuntimeError(msg)

		output_socket_methods = self.filtered_event_methods_by_event(
			ct.FlowEvent.OutputRequested,
			(output_socket_name, None, kind),
		)

		# Run (=1) Method
		if output_socket_methods:
			if len(output_socket_methods) > 1:
				msg = f'More than one method found for ({output_socket_name}, {kind.value!s}.'
				raise RuntimeError(msg)

			return output_socket_methods[0](self)

		# Auxiliary Fallbacks
		if optional or kind in [ct.FlowKind.Info, ct.FlowKind.Params]:
			return ct.FlowSignal.NoFlow

		msg = f'No output method for ({output_socket_name}, {kind})'
		raise ValueError(msg)

	####################
	# - Event Trigger
	####################
	def _should_recompute_output_socket(
		self,
		method_info: events.InfoOutputRequested,
		input_socket_name: ct.SocketName | None,
		input_socket_kinds: set[ct.FlowKind] | None,
		prop_name: str | None,
	) -> bool:
		return (
			prop_name is not None
			and prop_name in method_info.depon_props
			or input_socket_name is not None
			and (
				input_socket_name in method_info.depon_input_sockets
				and (
					input_socket_kinds is None
					or (
						isinstance(
							_kind := method_info.depon_input_socket_kinds.get(
								input_socket_name, ct.FlowKind.Value
							),
							set,
						)
						and input_socket_kinds.intersection(_kind)
					)
					or _kind == ct.FlowKind.Value
					or _kind in input_socket_kinds
				)
				or (
					method_info.depon_all_loose_input_sockets
					and input_socket_name in self.loose_input_sockets
				)
			)
		)

	@bl_cache.cached_bl_property()
	def _dependent_outputs(
		self,
	) -> dict[
		tuple[ct.SocketName, ct.FlowKind], set[tuple[ct.SocketName, ct.FlowKind]]
	]:
		## TODO: Cleanup
		## TODO: Detect cycles?
		## TODO: Networkx?
		altered_to_invalidated = defaultdict(set)
		output_requested_methods = self.event_methods_by_event[
			ct.FlowEvent.OutputRequested
		]

		for altered_method in output_requested_methods:
			altered_info = altered_method.callback_info
			altered_key = (altered_info.output_socket_name, altered_info.kind)

			for invalidated_method in output_requested_methods:
				invalidated_info = invalidated_method.callback_info

				if (
					altered_info.output_socket_name
					in invalidated_info.depon_output_sockets
				):
					is_same_kind = (
						altered_info.kind
						== (
							_kind := invalidated_info.depon_output_socket_kinds.get(
								altered_info.output_socket_name
							)
						)
						or (isinstance(_kind, set) and altered_info.kind in _kind)
						or altered_info.kind == ct.FlowKind.Value
					)

					if is_same_kind:
						invalidated_key = (
							invalidated_info.output_socket_name,
							invalidated_info.kind,
						)
						altered_to_invalidated[altered_key].add(invalidated_key)

		return altered_to_invalidated

	def trigger_event(
		self,
		event: ct.FlowEvent,
		socket_name: ct.SocketName | None = None,
		socket_kinds: set[ct.FlowKind] | None = None,
		prop_name: ct.SocketName | None = None,
	) -> None:
		"""Recursively triggers events forwards or backwards along the node tree, allowing nodes in the update path to react.

		Use `events` decorators to define methods that react to particular `ct.FlowEvent`s.

		Notes:
			This can be an unpredictably heavy function, depending on the node graph topology.

			Doesn't accept `LinkChanged` events; they are translated to `DataChanged` on the socket.
			This is on purpose: It seems to be a bad idea to try and differentiate between "changes in data" and "changes in linkage".

		Parameters:
			event: The event to report forwards/backwards along the node tree.
			socket_name: The input socket that was altered, if any, in order to trigger this event.
			pop_name: The property that was altered, if any, in order to trigger this event.
		"""
		log.debug(
			'%s: Triggered Event %s (socket_name=%s, socket_kinds=%s, prop_name=%s)',
			self.sim_node_name,
			event,
			str(socket_name),
			str(socket_kinds),
			str(prop_name),
		)
		# Outflow Socket Kinds
		## -> Something has happened!
		## -> The effect is yet to be determined...
		## -> We will watch for which kinds actually invalidate.
		## -> ...Then ONLY propagate kinds that have an invalidated outsck.
		## -> This way, kinds get "their own" invalidation chains.
		## -> ...While still respecting "crossovers".
		altered_socket_kinds = set()

		# Invalidate Caches on DataChanged
		if event == ct.FlowEvent.DataChanged:
			input_socket_name = socket_name  ## Trigger direction is forwards

			# Invalidate Input Socket Cache
			if input_socket_name is not None:
				if socket_kinds is None:
					self._compute_input.invalidate(
						input_socket_name=input_socket_name,
						kind=...,
						unit_system=...,
					)
				else:
					for socket_kind in socket_kinds:
						self._compute_input.invalidate(
							input_socket_name=input_socket_name,
							kind=socket_kind,
							unit_system=...,
						)

			# Invalidate Output Socket Cache
			for output_socket_method in self.event_methods_by_event[
				ct.FlowEvent.OutputRequested
			]:
				method_info = output_socket_method.callback_info
				if self._should_recompute_output_socket(
					method_info, socket_name, socket_kinds, prop_name
				):
					out_sckname = method_info.output_socket_name
					kind = method_info.kind

					# Invalidate Output Directly
					# log.critical(
					# '[%s] Invalidating: (%s, %s)',
					# self.sim_node_name,
					# out_sckname,
					# str(kind),
					# )
					altered_socket_kinds.add(kind)
					self.compute_output.invalidate(
						output_socket_name=out_sckname,
						kind=kind,
					)

					# Invalidate Any Dependent Outputs
					if (
						dep_outs := self._dependent_outputs.get((out_sckname, kind))
					) is not None:
						for dep_out in dep_outs:
							# log.critical(
							# '![%s] Invalidating: (%s, %s)',
							# self.sim_node_name,
							# dep_out[0],
							# dep_out[1],
							# )
							altered_socket_kinds.add(dep_out[1])
							self.compute_output.invalidate(
								output_socket_name=dep_out[0],
								kind=dep_out[1],
							)

		# Run Triggered Event Methods
		stop_propagation = False
		triggered_event_methods = self.filtered_event_methods_by_event(
			event, (socket_name, prop_name, None)
		)
		for event_method in triggered_event_methods:
			stop_propagation |= event_method.stop_propagation
			# log.critical(
			# '%s: Running %s',
			# self.sim_node_name,
			# str(event_method.callback_info),
			# )
			event_method(self)

		# Propagate Event to All Sockets in "Trigger Direction"
		## The trigger chain goes node/socket/node/socket/...
		if not stop_propagation:
			direc = ct.FlowEvent.flow_direction[event]
			triggered_sockets = self._bl_sockets(direc=direc)
			for bl_socket in triggered_sockets:
				if direc == 'output' and not bl_socket.is_linked:
					continue

				# log.critical(
				# '![%s] Propagating: (%s, %s)',
				# self.sim_node_name,
				# event,
				# altered_socket_kinds,
				# )
				bl_socket.trigger_event(event, socket_kinds=altered_socket_kinds)

	####################
	# - Property Event: On Update
	####################
	def on_prop_changed(self, prop_name: str) -> None:
		"""Report that a particular property has changed, which may cause certain caches to regenerate.

		Notes:
			Called by **all** valid `bpy.prop.Property` definitions in the addon, via their update methods.

			May be called in a threaded context - careful!

		Parameters:
			prop_name: The name of the property that changed.
		"""
		# All Attributes: Trigger Event
		## -> This declares that the single property has changed.
		## -> This should happen first, in case dependents need a cache.
		if hasattr(self, prop_name):
			self.trigger_event(ct.FlowEvent.DataChanged, prop_name=prop_name)

		# BLField Attributes: Invalidate BLField Dependents
		## -> Dependent props will generally also trigger on_prop_changed.
		## -> The recursion ends with the depschain.
		## -> WARNING: The chain is not checked for ex. cycles.
		if prop_name in self.blfields:
			self.invalidate_blfield_deps(prop_name)

	####################
	# - UI Methods
	####################
	def draw_buttons(
		self,
		context: bpy.types.Context,
		layout: bpy.types.UILayout,
	) -> None:
		"""Draws the UI of the node.

		- **Locked** (`self.locked`): The UI will be unusable.
		- **Active Preset** (`self.active_preset`): The preset selector will display.
		- **Active Socket Set** (`self.active_socket_set`): The socket set selector will display.
		- **Use Sim Node Name** (`self.use_sim_node_name`): The `self.sim_node_name` will display.
		- **Properties**: Node properties will display, if `self.draw_props()` is overridden.
		- **Operators**: Node operators will display, if `self.draw_operators()` is overridden.
		- **Info**: Node information will display, if `self.draw_info()` is overridden.

		Parameters:
			context: The current Blender context.
			layout: Target for defining UI elements.
		"""
		if self.locked:
			layout.enabled = False

		if self.active_socket_set:
			layout.prop(self, self.blfields['active_socket_set'], text='')

		if self.active_preset is not None:
			layout.prop(self, self.blfields['active_preset'], text='')

		# Draw Name
		if self.use_sim_node_name:
			row = layout.row(align=True)
			row.label(text='', icon='FILE_TEXT')
			row.prop(self, self.blfields['sim_node_name'], text='')

		# Draw Name
		self.draw_props(context, layout)
		self.draw_operators(context, layout)
		self.draw_info(context, layout)

	def draw_props(
		self, context: bpy.types.Context, layout: bpy.types.UILayout
	) -> None:
		"""Draws any properties of the node.

		Notes:
			Should be overriden by individual node classes, if they have properties to expose.

		Parameters:
			context: The current Blender context.
			layout: Target for defining UI elements.
		"""

	def draw_operators(
		self, context: bpy.types.Context, layout: bpy.types.UILayout
	) -> None:
		"""Draws any operators associated with the node.

		Notes:
			Should be overriden by individual node classes, if they have operators to expose.

		Parameters:
			context: The current Blender context.
			layout: Target for defining UI elements.
		"""

	def draw_info(self, context: bpy.types.Context, layout: bpy.types.UILayout) -> None:
		"""Draws any runtime information associated with the node.

		Notes:
			Should be overriden by individual node classes, if they have runtime information to show.

		Parameters:
			context: The current Blender context.
			layout: Target for defining UI elements.
		"""

	####################
	# - Blender Node Methods
	####################
	@classmethod
	def poll(cls, node_tree: bpy.types.NodeTree) -> bool:
		"""Render the node exlusively instantiable within a Maxwell Sim nodetree.

		Notes:
			Run by Blender when determining instantiability of a node.

		Parameters:
			node_tree: The node tree within which the instantiability of this node should be determined.

		Returns:
			Whether or not the node can be instantiated within the given node tree.
		"""
		return node_tree.bl_idname == ct.TreeType.MaxwellSim.value

	def init(self, _: bpy.types.Context) -> None:
		"""Initialize the node instance, including ID, name, socket, presets, and the execution of any `on_value_changed` methods with the `run_on_init` keyword set.

		Notes:
			Run by Blender when a new instance of a node is added to a tree.
		"""
		# Initialize Instance ID
		## -> This is used by various caches from 'bl_cache'.
		## -> Also generates (first-time) the various enums.
		self.reset_instance_id()

		# Initialize Sockets
		## -> Ensures the availability of static sockets before dynamic fields.
		self._sync_sockets()

		# Initialize Dynamic Field Persistance
		## -> Ensures the availability of enum items for subsequent setters.
		self.regenerate_dynamic_field_persistance()

		# Initialize Name
		## -> Ensures the availability of sim_node_name immediately.
		self.sim_node_name = self.name

		# Event Methods
		## Run any 'DataChanged' methods with 'run_on_init' set.
		## Semantically: Creating data _arguably_ changes it.
		## -> Compromise: Users explicitly say 'run_on_init' in @on_value_changed
		for event_method in [
			event_method
			for event_method in self.event_methods_by_event[ct.FlowEvent.DataChanged]
			if event_method.callback_info.run_on_init
		]:
			event_method(self)

	def update(self) -> None:
		"""Explicitly do nothing per-node on tree changes.

		For responding to changes affecting only this node, use decorators from `events`.

		Notes:
			Run by Blender on all node instances whenever anything **in the entire node tree** changes.
		"""

	def copy(self, _: bpy.types.Node) -> None:
		"""Generate a new instance ID and Sim Node Name on the duplicated node.

		Notes:
			Blender runs this when instantiating this node from an existing node.
		"""
		# Generate New Instance ID
		self.reset_instance_id()

		# Generate New Instance ID for Sockets
		## Sockets can't do this themselves.
		for bl_sockets in [self.inputs, self.outputs]:
			for bl_socket in bl_sockets:
				bl_socket.reset_instance_id()

		# Generate New Sim Node Name
		## -> Blender will adds .00# so that `self.name` is unique.
		## -> We can shamelessly piggyback on this for unique managed objs.
		## -> ...But to avoid stealing the old node's mobjs, we first rename.
		self.sim_node_name = self.name

		# Event Methods
		## -> Re-run any 'DataChanged' methods with 'run_on_init' set.
		## -> Copying a node ~ re-initializing the new node.
		for event_method in [
			event_method
			for event_method in self.event_methods_by_event[ct.FlowEvent.DataChanged]
			if event_method.callback_info.run_on_init
		]:
			event_method(self)

	def free(self) -> None:
		"""Cleans various instance-associated data up, so the node can be cleanly deleted.

		- **Locking**: The entire input chain will be unlocked. Since we can't prevent the deletion, this is one way to prevent "dangling locks".
		- **Managed Objects**: `.free()` will be run on all managed objects.
		- **`NodeLinkCache`**: `.on_node_removed(self)` will be run on the node tree, so it can correctly adjust the `NodeLinkCache`. **This is essential for avoiding "use-after-free" crashes.**
		- **Non-Persistent Cache**: `bl_cache.invalidate_nonpersist_instance_id` will be run, so that any caches indexed by the instance ID of the to-be-deleted node will be cleared away.

		Notes:
			Run by Blender **before** executing user-requested deletion of a node.
		"""
		node_tree = self.id_data

		# Unlock
		## This is one approach to the "deleted locked nodes" problem.
		## Essentially, deleting a locked node will unlock along input chain.
		## It also counts if any of the input sockets are linked and locked.
		## Thus, we prevent "dangling locks".
		if self.locked or any(
			bl_socket.is_linked and bl_socket.locked
			for bl_socket in self.inputs.values()
		):
			self.trigger_event(ct.FlowEvent.DisableLock)

		# Free Managed Objects
		for managed_obj in self.managed_objs.values():
			managed_obj.free()

		# Update NodeTree Caches
		## The NodeTree keeps caches to for optimized event triggering.
		## However, ex. deleted nodes also deletes links, without cache update.
		## By reporting that we're deleting the node, the cache stays happy.
		node_tree.on_node_removed(self)

		# Invalidate Non-Persistent Cache
		## Prevents memory leak due to dangling cache entries for deleted nodes.
		bl_cache.invalidate_nonpersist_instance_id(self.instance_id)
