"""Defines a special base class, `MaxwellSimNode`, from which all nodes inherit.

Attributes:
	MANDATORY_PROPS: Properties that must be defined on the `MaxwellSimNode`.
"""

import typing as typ
import uuid
from types import MappingProxyType

import bpy
import sympy as sp
import typing_extensions as typx

from blender_maxwell.utils import logger

from .. import bl_cache, sockets
from .. import contracts as ct
from .. import managed_objs as _managed_objs
from . import events
from . import presets as _presets

log = logger.get(__name__)

MANDATORY_PROPS: set[str] = {'node_type', 'bl_label'}


class MaxwellSimNode(bpy.types.Node):
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

	use_sim_node_name: bool = False
	## TODO: bl_description from first line of __doc__?

	# Sockets
	input_sockets: typ.ClassVar[dict[str, sockets.base.SocketDef]] = MappingProxyType(
		{}
	)
	output_sockets: typ.ClassVar[dict[str, sockets.base.SocketDef]] = MappingProxyType(
		{}
	)
	input_socket_sets: typ.ClassVar[dict[str, dict[str, sockets.base.SocketDef]]] = (
		MappingProxyType({})
	)
	output_socket_sets: typ.ClassVar[dict[str, dict[str, sockets.base.SocketDef]]] = (
		MappingProxyType({})
	)

	# Presets
	presets: typ.ClassVar[dict[str, dict[str, _presets.PresetDef]]] = MappingProxyType(
		{}
	)

	# Managed Objects
	managed_obj_types: typ.ClassVar[
		dict[ct.ManagedObjName, type[_managed_objs.ManagedObj]]
	] = MappingProxyType({})

	####################
	# - Class Methods
	####################
	@classmethod
	def _assert_attrs_valid(cls) -> None:
		"""Asserts that all mandatory attributes are defined on the class.

		The list of mandatory objects is sourced from `base.MANDATORY_PROPS`.

		Raises:
			ValueError: If a mandatory attribute defined in `base.MANDATORY_PROPS` is not defined on the class.
		"""
		for cls_attr in MANDATORY_PROPS:
			if not hasattr(cls, cls_attr):
				msg = f'Node class {cls} does not define mandatory attribute "{cls_attr}".'
				raise ValueError(msg)

	@classmethod
	def set_prop(
		cls,
		prop_name: str,
		prop: bpy.types.Property,
		no_update: bool = False,
		update_with_name: str | None = None,
		**kwargs,
	) -> None:
		"""Adds a Blender property to a class via `__annotations__`, so it initializes with any subclass.

		Notes:
			- Blender properties can't be set within `__init_subclass__` simply by adding attributes to the class; they must be added as type annotations.
			- Must be called **within** `__init_subclass__`.

		Parameters:
			name: The name of the property to set.
			prop: The `bpy.types.Property` to instantiate and attach..
			no_update: Don't attach a `self.sync_prop()` callback to the property's `update`.
		"""
		_update_with_name = prop_name if update_with_name is None else update_with_name
		extra_kwargs = (
			{
				'update': lambda self, context: self.sync_prop(
					_update_with_name, context
				),
			}
			if not no_update
			else {}
		)
		cls.__annotations__[prop_name] = prop(
			**kwargs,
			**extra_kwargs,
		)

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
		]
		event_methods_by_event = {event: [] for event in set(ct.FlowEvent)}
		for method in event_methods:
			event_methods_by_event[method.event].append(method)

		return event_methods_by_event

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
		cls._assert_attrs_valid()

		# Node Properties
		## Identifiers
		cls.bl_idname: str = str(cls.node_type.value)
		cls.set_prop('instance_id', bpy.props.StringProperty, no_update=True)
		cls.set_prop('sim_node_name', bpy.props.StringProperty, default='')

		## Special States
		cls.set_prop('preview_active', bpy.props.BoolProperty, default=False)
		cls.set_prop('locked', bpy.props.BoolProperty, no_update=True, default=False)

		## Event Method Callbacks
		cls.event_methods_by_event = cls._gather_event_methods()

		## Active Socket Set
		if len(cls.input_socket_sets) + len(cls.output_socket_sets) > 0:
			socket_set_names = cls.socket_set_names()
			cls.set_prop(
				'active_socket_set',
				bpy.props.EnumProperty,
				name='Active Socket Set',
				description='Selector of active sockets',
				items=[
					(socket_set_name, socket_set_name, socket_set_name)
					for socket_set_name in socket_set_names
				],
				default=socket_set_names[0],
			)
		else:
			cls.active_socket_set = None

		## Active Preset
		## TODO: Validate Presets
		if cls.presets:
			cls.set_prop(
				'active_preset',
				bpy.props.EnumProperty,
				name='Active Preset',
				description='The currently active preset',
				items=[
					(
						preset_name,
						preset_def.label,
						preset_def.description,
					)
					for preset_name, preset_def in cls.presets.items()
				],
				default=next(cls.presets.keys()),
			)
		else:
			cls.active_preset = None

	####################
	# - Events: Default
	####################
	@events.on_value_changed(
		prop_name='sim_node_name',
		props={'sim_node_name', 'managed_objs', 'managed_obj_types'},
	)
	def _on_sim_node_name_changed(self, props: dict):
		log.info(
			'Changed Sim Node Name of a "%s" to "%s" (self=%s)',
			self.bl_idname,
			self.sim_node_name,
			str(self),
		)

		# Set Name of Managed Objects
		for mobj in props['managed_objs'].values():
			mobj.name = props['sim_node_name']

	@events.on_value_changed(prop_name='active_socket_set')
	def _on_socket_set_changed(self):
		log.info(
			'Changed Sim Node Socket Set to "%s"',
			self.active_socket_set,
		)
		self._sync_sockets()

	@events.on_value_changed(
		prop_name='active_preset', props=['presets', 'active_preset']
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

	@events.on_show_preview()
	def _on_show_preview(self):
		node_tree = self.id_data
		node_tree.report_show_preview(self)
		# Set Preview to Active
		## Implicitly triggers any @on_value_changed for preview_active.
		if not self.preview_active:
			self.preview_active = True

	@events.on_value_changed(prop_name='preview_active', props={'preview_active'})
	def _on_preview_changed(self, props):
		if not props['preview_active']:
			for mobj in self.managed_objs.values():
				mobj.hide_preview()

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
	# - Loose Sockets w/Events
	####################
	loose_input_sockets: dict[str, sockets.base.SocketDef] = bl_cache.BLField({})
	loose_output_sockets: dict[str, sockets.base.SocketDef] = bl_cache.BLField({})

	@events.on_value_changed(prop_name={'loose_input_sockets', 'loose_output_sockets'})
	def _on_loose_sockets_changed(self):
		self._sync_sockets()

	####################
	# - Socket Accessors
	####################
	def _bl_sockets(
		self, direc: typx.Literal['input', 'output']
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
		direc: typx.Literal['input', 'output'],
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
		self, direc: typx.Literal['input', 'output']
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
				node_tree.on_node_socket_removed(bl_socket)
				all_bl_sockets.remove(bl_socket)

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
				bl_socket = all_bl_sockets.new(
					str(socket_def.socket_type.value),
					socket_name,
				)
				bl_socket.display_shape = bl_socket.socket_shape

				# Record Socket Creation
				created_sockets[socket_name] = socket_def

			# Initialize Just-Created BL Sockets
			for socket_name, socket_def in created_sockets.items():
				socket_def.init(all_bl_sockets[socket_name])

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
	# - Managed Objects
	####################
	@bl_cache.cached_bl_property(persist=True)
	def managed_objs(self) -> dict[str, _managed_objs.ManagedObj]:
		"""Access the managed objects defined on this node.

		Persistent cache ensures that the managed objects are only created on first access, even across file reloads.
		"""
		if self.managed_obj_types:
			return {
				mobj_name: mobj_type(self.sim_node_name)
				for mobj_name, mobj_type in self.managed_obj_types.items()
			}

		return {}

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
		if (bl_socket := self.inputs.get(input_socket_name)) is not None:
			return (
				ct.FlowKind.scale_to_unit_system(
					kind,
					bl_socket.compute_data(kind=kind),
					bl_socket.socket_type,
					unit_system,
				)
				if unit_system is not None
				else bl_socket.compute_data(kind=kind)
			)

		if optional:
			return None

		msg = f'Input socket "{input_socket_name}" on "{self.bl_idname}" is not an active input socket'
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

		msg = f'No output method for ({output_socket_name}, {kind.value!s}'
		raise ValueError(msg)

	####################
	# - Event Trigger
	####################
	def _should_recompute_output_socket(
		self,
		method_info: events.InfoOutputRequested,
		input_socket_name: ct.SocketName,
		prop_name: str,
	) -> bool:
		return (
			prop_name is not None
			and prop_name in method_info.depon_props
			or input_socket_name is not None
			and (
				input_socket_name in method_info.depon_input_sockets
				or (
					method_info.depon_all_loose_input_sockets
					and input_socket_name in self.loose_input_sockets
				)
			)
		)

	def trigger_event(
		self,
		event: ct.FlowEvent,
		socket_name: ct.SocketName | None = None,
		prop_name: ct.SocketName | None = None,
	) -> None:
		"""Recursively triggers events forwards or backwards along the node tree, allowing nodes in the update path to react.

		Use `events` decorators to define methods that react to particular `ct.FlowEvent`s.

		Notes:
			This can be an unpredictably heavy function, depending on the node graph topology.

		Parameters:
			event: The event to report forwards/backwards along the node tree.
			socket_name: The input socket that was altered, if any, in order to trigger this event.
			pop_name: The property that was altered, if any, in order to trigger this event.
		"""
		if event == ct.FlowEvent.DataChanged:
			input_socket_name = socket_name  ## Trigger direction is forwards

			# Invalidate Input Socket Cache
			if input_socket_name is not None:
				self._compute_input.invalidate(
					input_socket_name=input_socket_name,
					kind=...,
					unit_system=...,
				)

			# Invalidate Output Socket Cache
			for output_socket_method in self.event_methods_by_event[
				ct.FlowEvent.OutputRequested
			]:
				method_info = output_socket_method.callback_info
				if self._should_recompute_output_socket(
					method_info, socket_name, prop_name
				):
					self.compute_output.invalidate(
						output_socket_name=method_info.output_socket_name,
						kind=method_info.kind,
					)

		# Run Triggered Event Methods
		stop_propagation = False
		triggered_event_methods = self.filtered_event_methods_by_event(
			event, (socket_name, prop_name, None)
		)
		for event_method in triggered_event_methods:
			stop_propagation |= event_method.stop_propagation
			event_method(self)

		# Propagate Event to All Sockets in "Trigger Direction"
		## The trigger chain goes node/socket/node/socket/...
		if not stop_propagation:
			triggered_sockets = self._bl_sockets(
				direc=ct.FlowEvent.flow_direction[event]
			)
			for bl_socket in triggered_sockets:
				bl_socket.trigger_event(event)

	####################
	# - Property Event: On Update
	####################
	def sync_prop(self, prop_name: str, _: bpy.types.Context) -> None:
		"""Report that a particular property has changed, which may cause certain caches to regenerate.

		Notes:
			Called by **all** valid `bpy.prop.Property` definitions in the addon, via their update methods.

			May be called in a threaded context - careful!

		Parameters:
			prop_name: The name of the property that changed.
		"""
		if hasattr(self, prop_name):
			self.trigger_event(ct.FlowEvent.DataChanged, prop_name=prop_name)
		else:
			msg = f'Property {prop_name} not defined on node {self}'
			raise RuntimeError(msg)

	####################
	# - UI Methods
	####################
	def draw_buttons(
		self,
		context: bpy.types.Context,
		layout: bpy.types.UILayout,
	) -> None:
		"""Draws the UI of the node.

		- Locked (`self.locked`): The UI will be unusable.
		- Active Preset (`self.active_preset`): The preset selector will display.
		- Active Socket Set (`self.active_socket_set`): The socket set selector will display.
		- Use Sim Node Name (`self.use_sim_node_name`): The "Sim Node Name will display.
		- Properties (`self.draw_props()`): Node properties will display.
		- Operators (`self.draw_operators()`): Node operators will display.
		- Info (`self.draw_operators()`): Node information will display.

		Parameters:
			context: The current Blender context.
			layout: Target for defining UI elements.
		"""
		if self.locked:
			layout.enabled = False

		if self.active_socket_set:
			layout.prop(self, 'active_socket_set', text='')

		if self.active_preset is not None:
			layout.prop(self, 'active_preset', text='')

		# Draw Name
		if self.use_sim_node_name:
			row = layout.row(align=True)
			row.label(text='', icon='FILE_TEXT')
			row.prop(self, 'sim_node_name', text='')

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
		## This is used by various caches from 'bl_cache'.
		self.instance_id = str(uuid.uuid4())

		# Initialize Name
		## This is used whenever a unique name pointing to this node is needed.
		## Contrary to self.name, it can be altered by the user as a property.
		self.sim_node_name = self.name

		# Initialize Sockets
		## This initializes any nodes that need initializing
		self._sync_sockets()

		# Apply Preset
		## This applies the default preset, if any.
		if self.active_preset:
			self._on_active_preset_changed()

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
		self.instance_id = str(uuid.uuid4())

		# Generate New Sim Node Name
		## Blender will automatically add .001 so that `self.name` is unique.
		self.sim_node_name = self.name

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
