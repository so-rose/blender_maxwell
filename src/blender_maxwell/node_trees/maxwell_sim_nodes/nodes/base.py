import typing as typ
import uuid
from types import MappingProxyType

import bpy
import sympy as sp
import typing_extensions as typx

from ....utils import extra_sympy_units as spux
from ....utils import logger
from .. import bl_cache
from .. import contracts as ct
from .. import managed_objs as _managed_objs
from . import events

log = logger.get(__name__)

MANDATORY_PROPS = {'node_type', 'bl_label'}


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
	input_sockets: typ.ClassVar[dict[str, ct.schemas.SocketDef]] = MappingProxyType({})
	output_sockets: typ.ClassVar[dict[str, ct.schemas.SocketDef]] = MappingProxyType({})
	input_socket_sets: typ.ClassVar[dict[str, dict[str, ct.schemas.SocketDef]]] = (
		MappingProxyType({})
	)
	output_socket_sets: typ.ClassVar[dict[str, dict[str, ct.schemas.SocketDef]]] = (
		MappingProxyType({})
	)

	# Presets
	presets: typ.ClassVar = MappingProxyType({})

	# Managed Objects
	managed_obj_defs: typ.ClassVar[
		dict[ct.ManagedObjName, ct.schemas.ManagedObjDef]
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
		"""Gathers all methods called in response to actions/events observed by the node.

		Notes:
			- 'Event methods' must have an attribute 'action_type' in order to be picked up.
			- 'Event methods' must have an attribute 'action_type'.

		Returns:
			Event methods, indexed by the action that (maybe) triggers them.
		"""
		event_methods = [
			method
			for attr_name in dir(cls)
			if hasattr(method := getattr(cls, attr_name), 'action_type')
			and method.action_type in set(ct.DataFlowAction)
		]
		event_methods_by_action = {
			action_type: [] for action_type in set(ct.DataFlowAction)
		}
		for method in event_methods:
			event_methods_by_action[method.action_type].append(method)

		return event_methods_by_action

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
		cls.event_methods_by_action = cls._gather_event_methods()

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
	# - Events: Class Properties
	####################
	@events.on_value_changed(prop_name='active_socket_set')
	def _on_socket_set_changed(self):
		log.info(
			'Changed Sim Node Socket Set to "%s"',
			self.active_socket_set,
		)
		self._sync_sockets()

	@events.on_value_changed(
		prop_name='sim_node_name',
		props={'sim_node_name', 'managed_objs', 'managed_obj_defs'},
	)
	def _on_sim_node_name_changed(self, props: dict):
		log.info(
			'Changed Sim Node Name of a "%s" to "%s" (self=%s)',
			self.bl_idname,
			self.sim_node_name,
			str(self),
		)

		# Set Name of Managed Objects
		for mobj_id, mobj in props['managed_objs'].items():
			mobj_def = props['managed_obj_defs'][mobj_id]
			mobj.name = mobj_def.name_prefix + props['sim_node_name']

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

				## TODO: Account for DataFlowKind
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
				if isinstance(mobj, _managed_objs.ManagedBLMesh):
					## TODO: This is a Workaround
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
	# - Loose Sockets
	####################
	loose_input_sockets: dict[str, ct.schemas.SocketDef] = bl_cache.BLField({})
	loose_output_sockets: dict[str, ct.schemas.SocketDef] = bl_cache.BLField({})

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

		Note:
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
	) -> dict[ct.SocketName, ct.schemas.SocketDef]:
		"""Retrieve all socket definitions for sockets that should be defined, according to the `self.active_socket_set`.

		Note:
			You should probably use `self.active_socket_defs()`

		Parameters:
			direc: The direction to load Blender sockets from.

		Returns:
			Mapping from socket names to corresponding `ct.schemas.SocketDef`s.

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
	) -> dict[ct.SocketName, ct.schemas.SocketDef]:
		"""Retrieve all socket definitions for sockets that should be defined.

		Parameters:
			direc: The direction to load Blender sockets from.

		Returns:
			Mapping from socket names to corresponding `ct.schemas.SocketDef`s.
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

		Note:
			Must be called after any change to socket definitions, including loose
		sockets.
		"""
		self._prune_inactive_sockets()
		self._add_new_active_sockets()

	####################
	# - Managed Objects
	####################
	managed_bl_meshes: dict[str, _managed_objs.ManagedBLMesh] = bl_cache.BLField({})
	managed_bl_images: dict[str, _managed_objs.ManagedBLImage] = bl_cache.BLField({})
	managed_bl_modifiers: dict[str, _managed_objs.ManagedBLModifier] = bl_cache.BLField(
		{}
	)

	@bl_cache.cached_bl_property(
		persist=False
	)  ## Disable broken ManagedObj union DECODER
	def managed_objs(self) -> dict[str, _managed_objs.ManagedObj]:
		"""Access the managed objects defined on this node.

		Persistent cache ensures that the managed objects are only created on first access, even across file reloads.
		"""
		if self.managed_obj_defs:
			if not (
				managed_objs := (
					self.managed_bl_meshes
					| self.managed_bl_images
					| self.managed_bl_modifiers
				)
			):
				return {
					mobj_name: mobj_def.mk(mobj_def.name_prefix + self.sim_node_name)
					for mobj_name, mobj_def in self.managed_obj_defs.items()
				}
			return managed_objs

		return {}

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
		# col = layout.column(align=False)
		if self.use_sim_node_name:
			# row = col.row(align=True)
			row = layout.row(align=True)
			row.label(text='', icon='FILE_TEXT')
			row.prop(self, 'sim_node_name', text='')

		# Draw Name
		self.draw_props(context, layout)
		self.draw_operators(context, layout)
		self.draw_info(context, layout)
		# self.draw_props(context, col)
		# self.draw_operators(context, col)
		# self.draw_info(context, col)

	def draw_props(self, context, layout):
		pass

	def draw_operators(self, context, layout):
		pass

	def draw_info(self, context, layout):
		pass

	####################
	# - Special Compute Input / Output Caches
	####################

	## Compute Output Cache
	## -> KEY: output socket name, kind
	## -> INV: When DataChanged triggers with one of the event_method dependencies:
	##      - event_method.dependencies.input_sockets has DataChanged socket_name
	##      - event_method.dependencies.input_socket_kinds has DataChanged kind
	##      - DataChanged socket_name is loose and event_method wants all-loose
	##      - event_method.dependencies.props has DataChanged prop_name
	def _hit_cached_output_socket_value(
		self,
		compute_output_socket_cb: typ.Callable[[], typ.Any],
		output_socket_name: ct.SocketName,
		kind: ct.DataFlowKind,
	) -> typ.Any | None:
		"""Retrieve a cached output socket value by `output_socket_name, kind`."""
		# Create Non-Persistent Cache Entry
		if bl_cache.CACHE_NOPERSIST.get(self.instance_id) is None:
			bl_cache.CACHE_NOPERSIST[self.instance_id] = {}
		cache_nopersist = bl_cache.CACHE_NOPERSIST[self.instance_id]

		# Create Output Socket Cache Entry
		if cache_nopersist.get('_cached_output_sockets') is None:
			cache_nopersist['_cached_output_sockets'] = {}
		cached_output_sockets = cache_nopersist['_cached_output_sockets']

		# Try Hit on Cached Output Sockets
		cached_value = cached_output_sockets.get((output_socket_name, kind))
		if cached_value is None:
			value = compute_output_socket_cb()
			cached_output_sockets[(output_socket_name, kind)] = value
		else:
			value = cached_value

		return value

	def _invalidate_cached_output_socket_value(
		self, output_socket_name: ct.SocketName, kind: ct.DataFlowKind
	) -> None:
		# Create Non-Persistent Cache Entry
		if bl_cache.CACHE_NOPERSIST.get(self.instance_id) is None:
			return
		cache_nopersist = bl_cache.CACHE_NOPERSIST[self.instance_id]

		# Create Output Socket Cache Entry
		if cache_nopersist.get('_cached_output_sockets') is None:
			return
		cached_output_sockets = cache_nopersist['_cached_output_sockets']

		# Try Hit & Delete
		cached_output_sockets.pop((output_socket_name, kind), None)

	## Input Cache
	## -> KEY: input socket name, kind, unit system
	## -> INV: DataChanged w/socket name
	def _hit_cached_input_socket_value(
		self,
		compute_input_socket_cb: typ.Callable[[typ.Self], typ.Any],
		input_socket_name: ct.SocketName,
		kind: ct.DataFlowKind,
		unit_system: dict[ct.SocketType, sp.Expr],
	) -> typ.Any | None:
		# Create Non-Persistent Cache Entry
		if bl_cache.CACHE_NOPERSIST.get(self.instance_id) is None:
			bl_cache.CACHE_NOPERSIST[self.instance_id] = {}
		cache_nopersist = bl_cache.CACHE_NOPERSIST[self.instance_id]

		# Create Output Socket Cache Entry
		if cache_nopersist.get('_cached_input_sockets') is None:
			cache_nopersist['_cached_input_sockets'] = {}
		cached_input_sockets = cache_nopersist['_cached_input_sockets']

		# Try Hit on Cached Output Sockets
		encoded_unit_system = bl_cache.ENCODER.encode(unit_system).decode('utf-8')
		cached_value = cached_input_sockets.get(
			(input_socket_name, kind, encoded_unit_system),
		)
		if cached_value is None:
			value = compute_input_socket_cb()
			cached_input_sockets[(input_socket_name, kind, encoded_unit_system)] = value
		else:
			value = cached_value
		return value

	def _invalidate_cached_input_socket_value(
		self,
		input_socket_name: ct.SocketName,
	) -> None:
		# Create Non-Persistent Cache Entry
		if bl_cache.CACHE_NOPERSIST.get(self.instance_id) is None:
			return
		cache_nopersist = bl_cache.CACHE_NOPERSIST[self.instance_id]

		# Create Output Socket Cache Entry
		if cache_nopersist.get('_cached_input_sockets') is None:
			return
		cached_input_sockets = cache_nopersist['_cached_input_sockets']

		# Try Hit & Delete
		for cached_input_socket in list(cached_input_sockets.keys()):
			if cached_input_socket[0] == input_socket_name:
				cached_input_sockets.pop(cached_input_socket, None)

	####################
	# - Data Flow
	####################
	## TODO: Lazy input socket list in events.py callbacks, to replace random scattered `_compute_input` calls.
	def _compute_input(
		self,
		input_socket_name: ct.SocketName,
		kind: ct.DataFlowKind = ct.DataFlowKind.Value,
		unit_system: dict[ct.SocketType, sp.Expr] | None = None,
		optional: bool = False,
	) -> typ.Any:
		"""Computes the data of an input socket, following links if needed.

		Note:
			The semantics derive entirely from `sockets.MaxwellSimSocket.compute_data()`.

		Parameters:
			input_socket_name: The name of the input socket to compute the value of.
				It must be currently active.
			kind: The data flow kind to compute.
		"""
		if (bl_socket := self.inputs.get(input_socket_name)) is not None:
			return self._hit_cached_input_socket_value(
				lambda: (
					ct.DataFlowKind.scale_to_unit_system(
						kind,
						bl_socket.compute_data(kind=kind),
						bl_socket.socket_type,
						unit_system,
					)
					if unit_system is not None
					else bl_socket.compute_data(kind=kind)
				),
				input_socket_name,
				kind,
				unit_system,
			)
		if optional:
			return None

		msg = f'Input socket "{input_socket_name}" on "{self.bl_idname}" is not an active input socket'
		raise ValueError(msg)

	def compute_output(
		self,
		output_socket_name: ct.SocketName,
		kind: ct.DataFlowKind = ct.DataFlowKind.Value,
		optional: bool = False,
	) -> typ.Any:
		"""Computes the value of an output socket.

		Parameters:
			output_socket_name: The name declaring the output socket, for which this method computes the output.
			kind: The DataFlowKind to use when computing the output socket value.

		Returns:
			The value of the output socket, as computed by the dedicated method
			registered using the `@computes_output_socket` decorator.
		"""
		if self.outputs.get(output_socket_name) is None:
			if optional:
				return None
			msg = f"Can't compute nonexistent output socket name {output_socket_name}, as it's not currently active"
			raise RuntimeError(msg)

		output_socket_methods = self.event_methods_by_action[
			ct.DataFlowAction.OutputRequested
		]
		possible_output_socket_methods = [
			output_socket_method
			for output_socket_method in output_socket_methods
			if kind == output_socket_method.callback_info.kind
			and (
				output_socket_name
				== output_socket_method.callback_info.output_socket_name
				or (
					output_socket_method.callback_info.any_loose_output_socket
					and output_socket_name in self.loose_output_sockets
				)
			)
		]
		if len(possible_output_socket_methods) == 1:
			return self._hit_cached_output_socket_value(
				lambda: possible_output_socket_methods[0](self),
				output_socket_name,
				kind,
			)
			return possible_output_socket_methods[0](self)

		if len(possible_output_socket_methods) == 0:
			msg = f'No output method for ({output_socket_name}, {kind.value!s}'
			raise ValueError(msg)

		if len(possible_output_socket_methods) > 1:
			msg = (
				f'More than one method found for ({output_socket_name}, {kind.value!s}.'
			)
			raise RuntimeError(msg)

		msg = 'Somehow, a length is negative. Call NASA.'
		raise SystemError(msg)

	####################
	# - Action Chain
	####################
	def sync_prop(self, prop_name: str, _: bpy.types.Context) -> None:
		"""Report that a particular property has changed, which may cause certain caches to regenerate.

		Note:
			Called by **all** valid `bpy.prop.Property` definitions in the addon, via their update methods.

			May be called in a threaded context - careful!

		Parameters:
			prop_name: The name of the property that changed.
		"""
		if hasattr(self, prop_name):
			self.trigger_action(ct.DataFlowAction.DataChanged, prop_name=prop_name)
		else:
			msg = f'Property {prop_name} not defined on node {self}'
			raise RuntimeError(msg)

	@bl_cache.cached_bl_property(persist=False)
	def event_method_filter_by_action(self) -> dict[ct.DataFlowAction, typ.Callable]:
		"""Compute a map of DataFlowActions, to a function that filters its event methods.

		The filter expression may use attributes of `self`, or return `True` if no filtering should occur, or return `False` if methods should never run.
		"""
		return {
			ct.DataFlowAction.EnableLock: lambda *_: True,
			ct.DataFlowAction.DisableLock: lambda *_: True,
			ct.DataFlowAction.DataChanged: lambda event_method,
			socket_name,
			prop_name: (
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
			ct.DataFlowAction.OutputRequested: lambda *_: False,
			ct.DataFlowAction.ShowPreview: lambda *_: True,
			ct.DataFlowAction.ShowPlot: lambda *_: True,
		}

	def trigger_action(
		self,
		action: ct.DataFlowAction,
		socket_name: ct.SocketName | None = None,
		prop_name: ct.SocketName | None = None,
	) -> None:
		"""Recursively triggers actions/events forwards or backwards along the node tree, allowing nodes in the update path to react.

		Use `events` decorators to define methods that react to particular `ct.DataFlowAction`s.

		Note:
			This can be an unpredictably heavy function, depending on the node graph topology.

		Parameters:
			action: The action/event to report forwards/backwards along the node tree.
			socket_name: The input socket that was altered, if any, in order to trigger this event.
			pop_name: The property that was altered, if any, in order to trigger this event.
		"""
		if action == ct.DataFlowAction.DataChanged:
			# Invalidate Input/Output Socket Caches
			all_output_method_infos = [
				event_method.callback_info
				for event_method in self.event_methods_by_action[
					ct.DataFlowAction.OutputRequested
				]
			]
			input_sockets_to_invalidate_cached_values_of = set()
			output_sockets_to_invalidate_cached_values_of = set()

			# Invalidate by Dependent Input Socket
			if socket_name is not None:
				input_sockets_to_invalidate_cached_values_of.add(socket_name)

				## Output Socket: Invalidate if an Output Method Depends on Us
				output_sockets_to_invalidate_cached_values_of |= {
					(method_info.output_socket_name, method_info.kind)
					for method_info in all_output_method_infos
					if socket_name in method_info.depon_input_sockets
					or (
						socket_name in self.loose_input_sockets
						and method_info.depon_all_loose_input_sockets
					)
				}

			# Invalidate by Dependent Property
			if prop_name is not None:
				output_sockets_to_invalidate_cached_values_of |= {
					(method_info.output_socket_name, method_info.kind)
					for method_info in all_output_method_infos
					if prop_name in method_info.depon_props
				}

			# Invalidate Output Socket Values
			for key in input_sockets_to_invalidate_cached_values_of:
				# log.debug('Invalidating Input Socket Cache: %s', key)
				self._invalidate_cached_input_socket_value(key)

			for key in output_sockets_to_invalidate_cached_values_of:
				# log.debug('Invalidating Output Socket Cache: %s', key)
				self._invalidate_cached_output_socket_value(*key)

		# Run Triggered Event Methods
		stop_propagation = False  ## A method wants us to not continue
		event_methods_to_run = [
			event_method
			for event_method in self.event_methods_by_action[action]
			if self.event_method_filter_by_action[action](
				event_method, socket_name, prop_name
			)
		]
		for event_method in event_methods_to_run:
			stop_propagation |= event_method.stop_propagation
			event_method(self)

		# Trigger Action on Input/Output Sockets
		## The trigger chain goes node/socket/node/socket/...
		if (
			ct.DataFlowAction.stop_if_no_event_methods(action)
			and len(event_methods_to_run) == 0
		):
			return
		if not stop_propagation:
			triggered_sockets = self._bl_sockets(
				direc=ct.DataFlowAction.trigger_direction(action)
			)
			for bl_socket in triggered_sockets:
				bl_socket.trigger_action(action)

	####################
	# - Blender Node Methods
	####################
	@classmethod
	def poll(cls, node_tree: bpy.types.NodeTree) -> bool:
		"""Run (by Blender) to determine instantiability.

		Restricted to the MaxwellSimTreeType.
		"""
		return node_tree.bl_idname == ct.TreeType.MaxwellSim.value

	def init(self, context: bpy.types.Context):
		"""Run (by Blender) on node creation."""
		# Initialize Cache and Instance ID
		self.instance_id = str(uuid.uuid4())

		# Initialize Name
		self.sim_node_name = self.name
		## Only shown in draw_buttons if 'self.use_sim_node_name'

		# Initialize Sockets
		self._sync_sockets()

		# Apply Default Preset
		if self.active_preset:
			self._on_active_preset_changed()

		# Event Methods
		## Run any 'DataChanged' methods with 'run_on_init' set.
		## Semantically: Creating data _arguably_ changes it.
		## -> Compromise: Users explicitly say 'run_on_init' in @on_value_changed
		for event_method in [
			event_method
			for event_method in self.event_methods_by_action[
				ct.DataFlowAction.DataChanged
			]
			if event_method.callback_info.run_on_init
		]:
			event_method(self)

	def update(self) -> None:
		pass

	def copy(self, _: bpy.types.Node) -> None:
		"""Generate a new instance ID and Sim Node Name.

		Note:
			Blender runs this when instantiating this node from an existing node.

		Parameters:
			node: The existing node from which this node was copied.
		"""
		# Generate New Instance ID
		self.instance_id = str(uuid.uuid4())

		# Generate New Name
		## Blender will automatically add .001 so that `self.name` is unique.
		self.sim_node_name = self.name

	def free(self) -> None:
		"""Run (by Blender) when deleting the node."""
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
			self.trigger_action(ct.DataFlowAction.DisableLock)

		# Free Managed Objects
		for managed_obj in self.managed_objs.values():
			managed_obj.free()

		# Update NodeTree Caches
		## The NodeTree keeps caches to for optimized event triggering.
		## However, ex. deleted nodes also deletes links, without cache update.
		## By reporting that we're deleting the node, the cache stays happy.
		node_tree.sync_node_removed(self)

		# Invalidate Non-Persistent Cache
		## Prevents memory leak due to dangling cache entries for deleted nodes.
		bl_cache.invalidate_nonpersist_instance_id(self.instance_id)
