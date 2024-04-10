import json
import typing as typ
import uuid

import bpy
import pydantic as pyd
import typing_extensions as typx

from ....utils import logger
from .. import contracts as ct
from .. import sockets

log = logger.get(__name__)

CACHE: dict[str, typ.Any] = {}  ## By Instance UUID
## NOTE: CACHE does not persist between file loads.

_DEFAULT_LOOSE_SOCKET_SER = json.dumps(
	{
		'socket_names': [],
		'socket_def_names': [],
		'models': [],
	}
)  ## TODO: What in the jesus christ is this


class MaxwellSimNode(bpy.types.Node):
	# Fundamentals
	node_type: ct.NodeType
	bl_idname: str
	use_sim_node_name: bool = False
	bl_label: str
	# draw_label(self) -> str: pass

	# Style
	bl_description: str = ''

	# bl_width_default: float = 0.0
	# bl_width_min: float = 0.0
	# bl_width_max: float = 0.0

	# Sockets
	_output_socket_methods: dict

	input_sockets: typ.ClassVar[dict[str, ct.schemas.SocketDef]] = {}
	output_sockets: typ.ClassVar[dict[str, ct.schemas.SocketDef]] = {}
	input_socket_sets: typ.ClassVar[dict[str, dict[str, ct.schemas.SocketDef]]] = {}
	output_socket_sets: typ.ClassVar[dict[str, dict[str, ct.schemas.SocketDef]]] = {}

	# Presets
	presets: typ.ClassVar = {}

	# Managed Objects
	managed_obj_defs: typ.ClassVar[
		dict[ct.ManagedObjName, ct.schemas.ManagedObjDef]
	] = {}

	####################
	# - Initialization
	####################
	def __init_subclass__(cls, **kwargs: typ.Any):
		super().__init_subclass__(**kwargs)
		log.debug('Initializing Node: %s', cls.node_type)

		# Setup Blender ID for Node
		if not hasattr(cls, 'node_type'):
			msg = f"Node class {cls} does not define 'node_type', or it is does not have the type {ct.NodeType}"
			raise ValueError(msg)
		cls.bl_idname = str(cls.node_type.value)

		# Setup Instance ID for Node
		cls.__annotations__['instance_id'] = bpy.props.StringProperty(
			name='Instance ID',
			description='The instance ID of a particular MaxwellSimNode instance, used to index caches',
			default='',
		)

		# Setup Name Property for Node
		cls.__annotations__['sim_node_name'] = bpy.props.StringProperty(
			name='Sim Node Name',
			description='The name of a particular MaxwellSimNode node, which can be used to help identify data managed by the node',
			default='',
			update=(lambda self, context: self.sync_sim_node_name(context)),
		)

		# Setup "Previewing" Property for Node
		cls.__annotations__['preview_active'] = bpy.props.BoolProperty(
			name='Preview Active',
			description='Whether the preview (if any) is currently active',
			default=False,
			update=lambda self, context: self.sync_preview_active(context),
		)

		# Setup Locked Property for Node
		cls.__annotations__['locked'] = bpy.props.BoolProperty(
			name='Locked State',
			description="The lock-state of a particular MaxwellSimNode instance, which determines the node's user editability",
			default=False,
		)

		# Setup Blender Label for Node
		if not hasattr(cls, 'bl_label'):
			msg = f"Node class {cls} does not define 'bl_label'"
			raise ValueError(msg)

		# Setup Callback Methods
		cls._output_socket_methods = {
			method
			for attr_name in dir(cls)
			if hasattr(method := getattr(cls, attr_name), 'action_type')
			and method.action_type == 'computes_output_socket'
		}
		cls._on_value_changed_methods = {
			method
			for attr_name in dir(cls)
			if hasattr(method := getattr(cls, attr_name), 'action_type')
			and method.action_type == 'on_value_changed'
		}
		cls._on_show_plot = {
			method
			for attr_name in dir(cls)
			if hasattr(method := getattr(cls, attr_name), 'action_type')
			and method.action_type == 'on_show_plot'
		}
		cls._on_init = {
			method
			for attr_name in dir(cls)
			if hasattr(method := getattr(cls, attr_name), 'action_type')
			and method.action_type == 'on_init'
		}

		# Setup Socket Set Dropdown
		if not len(cls.input_socket_sets) + len(cls.output_socket_sets) > 0:
			cls.active_socket_set = None
		else:
			## Add Active Socket Set Enum
			socket_set_names = (
				_input_socket_set_names := list(cls.input_socket_sets.keys())
			) + [
				output_socket_set_name
				for output_socket_set_name in cls.output_socket_sets
				if output_socket_set_name not in _input_socket_set_names
			]
			socket_set_ids = [
				socket_set_name.replace(' ', '_').upper()
				for socket_set_name in socket_set_names
			]
			## TODO: Better deriv. of sock.set. ID, ex. ( is currently invalid.

			## Add Active Socket Set Enum
			cls.__annotations__['active_socket_set'] = bpy.props.EnumProperty(
				name='Active Socket Set',
				description='The active socket set',
				items=[
					(
						socket_set_name,
						socket_set_name,
						socket_set_name,
					)
					for socket_set_id, socket_set_name in zip(
						socket_set_ids,
						socket_set_names,
						strict=False,
					)
				],
				default=socket_set_names[0],
				update=lambda self, context: self.sync_active_socket_set(context),
			)

		# Setup Preset Dropdown
		if not cls.presets:
			cls.active_preset = None
		else:
			## TODO: Check that presets are represented in a socket that is guaranteed to be always available, specifically either a static socket or ALL static socket sets.
			cls.__annotations__['active_preset'] = bpy.props.EnumProperty(
				name='Active Preset',
				description='The active preset',
				items=[
					(
						preset_name,
						preset_def.label,
						preset_def.description,
					)
					for preset_name, preset_def in cls.presets.items()
				],
				default=next(cls.presets.keys()),
				update=lambda self, _: (self.sync_active_preset()()),
			)

	####################
	# - Generic Properties
	####################
	def sync_active_socket_set(self, context):
		self.sync_sockets()
		self.sync_prop('active_socket_set', context)

	def sync_sim_node_name(self, _):
		if (mobjs := CACHE[self.instance_id].get('managed_objs')) is None:
			return

		for mobj_id, mobj in mobjs.items():
			# Retrieve Managed Obj Definition
			mobj_def = self.managed_obj_defs[mobj_id]

			# Set Managed Obj Name
			mobj.name = mobj_def.name_prefix + self.sim_node_name
			## ManagedObj is allowed to alter the name when setting it.
			## - This will happen whenever the name is taken.
			## - If altered, set the 'sim_node_name' to the altered name.
			## - This will cause recursion, but only once.

	def sync_preview_active(self, _: bpy.types.Context):
		log.info(
			'Changed Preview Active in "%s" to "%s"',
			self.name,
			self.preview_active,
		)
		for method in self._on_value_changed_methods:
			if 'preview_active' in method.extra_data['changed_props']:
				log.info(
					'Running Previewer Callback "%s" in "%s")',
					method.__name__,
					self.name,
				)
				method(self)

	####################
	# - Managed Object Properties
	####################
	@property
	def managed_objs(self):
		if not CACHE.get(self.instance_id):
			CACHE[self.instance_id] = {}

		# If No Managed Objects in CACHE: Initialize Managed Objects
		## - This happens on every ex. file load, init(), etc. .
		## - ManagedObjects MUST the same object by name.
		## - We sync our 'sim_node_name' with all managed objects.
		## - (There is also a class-defined 'name_prefix' to differentiate)
		## - See the 'sim_node_name' w/its sync function.
		if CACHE[self.instance_id].get('managed_objs') is None:
			# Initialize the Managed Object Instance Cache
			CACHE[self.instance_id]['managed_objs'] = {}

			# Fill w/Managed Objects by Name Socket
			for mobj_id, mobj_def in self.managed_obj_defs.items():
				name = mobj_def.name_prefix + self.sim_node_name
				CACHE[self.instance_id]['managed_objs'][mobj_id] = mobj_def.mk(name)

			return CACHE[self.instance_id]['managed_objs']

		return CACHE[self.instance_id]['managed_objs']

	####################
	# - Socket Properties
	####################
	def active_bl_sockets(self, direc: typx.Literal['input', 'output']):
		return self.inputs if direc == 'input' else self.outputs

	def active_socket_set_sockets(
		self,
		direc: typx.Literal['input', 'output'],
	) -> dict:
		# No Active Socket Set: Return Nothing
		if not self.active_socket_set:
			return {}

		# Retrieve Active Socket Set Sockets
		socket_sets = (
			self.input_socket_sets if direc == 'input' else self.output_socket_sets
		)
		active_socket_set_sockets = socket_sets.get(self.active_socket_set)

		# Return Active Socket Set Sockets (if any)
		if not active_socket_set_sockets:
			return {}
		return active_socket_set_sockets

	def active_sockets(self, direc: typx.Literal['input', 'output']):
		static_sockets = self.input_sockets if direc == 'input' else self.output_sockets
		loose_sockets = (
			self.loose_input_sockets if direc == 'input' else self.loose_output_sockets
		)

		return (
			static_sockets | self.active_socket_set_sockets(direc=direc) | loose_sockets
		)

	####################
	# - Loose Sockets
	####################
	# Loose Sockets
	## Only Blender props persist as instance data
	ser_loose_input_sockets: bpy.props.StringProperty(
		name='Serialized Loose Input Sockets',
		description='JSON-serialized representation of loose input sockets.',
		default=_DEFAULT_LOOSE_SOCKET_SER,
	)
	ser_loose_output_sockets: bpy.props.StringProperty(
		name='Serialized Loose Input Sockets',
		description='JSON-serialized representation of loose input sockets.',
		default=_DEFAULT_LOOSE_SOCKET_SER,
	)

	## Internal Serialization/Deserialization Methods (yuck)
	def _ser_loose_sockets(self, deser: dict[str, ct.schemas.SocketDef]) -> str:
		if not all(isinstance(model, pyd.BaseModel) for model in deser.values()):
			msg = 'Trying to deserialize loose sockets with invalid SocketDefs (they must be `pydantic` BaseModels).'
			raise ValueError(msg)

		return json.dumps(
			{
				'socket_names': list(deser.keys()),
				'socket_def_names': [
					model.__class__.__name__ for model in deser.values()
				],
				'models': [
					model.model_dump()
					for model in deser.values()
					if isinstance(model, pyd.BaseModel)
				],
			}
		)  ## Big reliance on order-preservation of dicts here.)

	def _deser_loose_sockets(self, ser: str) -> dict[str, ct.schemas.SocketDef]:
		semi_deser = json.loads(ser)
		return {
			socket_name: getattr(sockets, socket_def_name)(**model_kwargs)
			for socket_name, socket_def_name, model_kwargs in zip(
				semi_deser['socket_names'],
				semi_deser['socket_def_names'],
				semi_deser['models'],
				strict=False,
			)
			if hasattr(sockets, socket_def_name)
		}

	@property
	def loose_input_sockets(self) -> dict[str, ct.schemas.SocketDef]:
		return self._deser_loose_sockets(self.ser_loose_input_sockets)

	@property
	def loose_output_sockets(self) -> dict[str, ct.schemas.SocketDef]:
		return self._deser_loose_sockets(self.ser_loose_output_sockets)

	## TODO: Some caching may play a role if this is all too slow.

	@loose_input_sockets.setter
	def loose_input_sockets(
		self,
		value: dict[str, ct.schemas.SocketDef],
	) -> None:
		# Prune Loose Sockets
		self.ser_loose_input_sockets = _DEFAULT_LOOSE_SOCKET_SER
		self.sync_sockets()

		# Install New Sockets
		if not value:
			self.ser_loose_input_sockets = _DEFAULT_LOOSE_SOCKET_SER
		else:
			self.ser_loose_input_sockets = self._ser_loose_sockets(value)

		# Synchronize Sockets
		self.sync_sockets()

	@loose_output_sockets.setter
	def loose_output_sockets(
		self,
		value: dict[str, ct.schemas.SocketDef],
	) -> None:
		# Prune Loose Sockets
		self.ser_loose_output_sockets = _DEFAULT_LOOSE_SOCKET_SER
		self.sync_sockets()

		# Install New Sockets
		if not value:
			self.ser_loose_output_sockets = _DEFAULT_LOOSE_SOCKET_SER
		else:
			self.ser_loose_output_sockets = self._ser_loose_sockets(value)

		# Synchronize Sockets
		self.sync_sockets()

	####################
	# - Socket Management
	####################
	def _prune_inactive_sockets(self):
		"""Remove all inactive sockets from the node.

		**NOTE**: Socket names must be unique within direction, active socket set, and loose socket set.
		"""
		for direc in ['input', 'output']:
			sockets = self.active_sockets(direc)
			bl_sockets = self.active_bl_sockets(direc)

			# Determine Sockets to Remove
			bl_sockets_to_remove = [
				bl_socket
				for socket_name, bl_socket in bl_sockets.items()
				if socket_name not in sockets
			]

			# Remove Sockets
			for bl_socket in bl_sockets_to_remove:
				bl_sockets.remove(bl_socket)

	def _add_new_active_sockets(self):
		"""Add and initialize all non-existing active sockets to the node.

		Existing sockets within the given direction are not re-created.
		"""
		for direc in ['input', 'output']:
			sockets = self.active_sockets(direc)
			bl_sockets = self.active_bl_sockets(direc)

			# Define BL Sockets
			created_sockets = {}
			for socket_name, socket_def in sockets.items():
				# Skip Existing Sockets
				if socket_name in bl_sockets:
					continue

				# Create BL Socket from Socket
				bl_socket = bl_sockets.new(
					str(socket_def.socket_type.value),
					socket_name,
				)
				bl_socket.display_shape = bl_socket.socket_shape
				## `display_shape` needs to be dynamically set

				# Record Created Socket
				created_sockets[socket_name] = socket_def

			# Initialize Just-Created BL Sockets
			for socket_name, socket_def in created_sockets.items():
				socket_def.init(bl_sockets[socket_name])

	def sync_sockets(self) -> None:
		"""Synchronize the node's sockets with the active sockets.

		- Any non-existing active socket will be added and initialized.
		- Any existing active socket will not be changed.
		- Any existing inactive socket will be removed.

		Must be called after any change to socket definitions, including loose
		sockets.
		"""
		self._prune_inactive_sockets()
		self._add_new_active_sockets()

	####################
	# - Preset Management
	####################
	def sync_active_preset(self) -> None:
		"""Applies the active preset by overwriting the value of preset-defined input sockets."""
		if not (preset_def := self.presets.get(self.active_preset)):
			msg = f'Tried to apply active preset, but the active preset ({self.active_preset}) is not in presets ({self.presets})'
			raise RuntimeError(msg)

		for socket_name, socket_value in preset_def.values.items():
			if not (bl_socket := self.inputs.get(socket_name)):
				msg = f'Tried to set preset socket/value pair ({socket_name}={socket_value}), but socket is not in active input sockets ({self.inputs})'
				raise ValueError(msg)

			bl_socket.value = socket_value
			## TODO: Lazy-valued presets?

	####################
	# - UI Methods
	####################
	def draw_buttons(
		self,
		context: bpy.types.Context,
		layout: bpy.types.UILayout,
	) -> None:
		if self.locked:
			layout.enabled = False

		if self.active_preset:
			layout.prop(self, 'active_preset', text='')

		if self.active_socket_set:
			layout.prop(self, 'active_socket_set', text='')

		# Draw Name
		col = layout.column(align=False)
		if self.use_sim_node_name:
			row = col.row(align=True)
			row.label(text='', icon='FILE_TEXT')
			row.prop(self, 'sim_node_name', text='')

		# Draw Name
		self.draw_props(context, col)
		self.draw_operators(context, col)
		self.draw_info(context, col)

	## TODO: Managed Operators instead of this shit
	def draw_props(self, context, layout):
		pass

	def draw_operators(self, context, layout):
		pass

	def draw_info(self, context, layout):
		pass

	def draw_buttons_ext(self, context, layout):
		pass

	## TODO: Side panel buttons for fanciness.

	def draw_plot_settings(self, _: bpy.types.Context, layout: bpy.types.UILayout):
		if self.locked:
			layout.enabled = False

	####################
	# - Data Flow
	####################
	def _compute_input(
		self,
		input_socket_name: ct.SocketName,
		kind: ct.DataFlowKind = ct.DataFlowKind.Value,
	) -> typ.Any | None:
		"""Computes the data of an input socket, by socket name and data flow kind, by asking the socket nicely via `bl_socket.compute_data`.

		Args:
			input_socket_name: The name of the input socket, as defined in `self.input_sockets`.
			kind: The kind of data flow to compute.
		"""
		if bl_socket := self.inputs.get(input_socket_name):
			return bl_socket.compute_data(kind=kind)

		msg = f'Input socket "{input_socket_name}" on "{self.bl_idname}" is not an active input socket'
		raise ValueError(msg)

	def compute_output(
		self,
		output_socket_name: ct.SocketName,
		kind: ct.DataFlowKind = ct.DataFlowKind.Value,
	) -> typ.Any:
		"""Computes the value of an output socket name, from its socket name.

		Searches methods decorated with `@computes_output_socket(output_socket_name, kind=..., ...)`, for a perfect match to the pair `socket_name..kind`.
		This method is run to produce the value.

		Args:
			output_socket_name: The name declaring the output socket, for which this method computes the output.
			kind: The DataFlowKind to use when computing the output socket value.

		Returns:
			The value of the output socket, as computed by the dedicated method
			registered using the `@computes_output_socket` decorator.
		"""
		possible_output_socket_methods = [
			output_socket_method
			for output_socket_method in self._output_socket_methods
			if kind == output_socket_method.extra_data['kind']
			and (
				output_socket_name
				== output_socket_method.extra_data['output_socket_name']
				or (
					output_socket_method.extra_data['any_loose_output_socket']
					and output_socket_name in self.loose_output_sockets
				)
			)
		]
		if len(possible_output_socket_methods) == 1:
			return possible_output_socket_methods[0](self)

		msg = f'No output method for ({output_socket_name}, {kind.value!s}'
		raise ValueError(msg)

	####################
	# - Action Chain
	####################
	def sync_prop(self, prop_name: str, _: bpy.types.Context):
		"""Called when a property has been updated."""
		if not hasattr(self, prop_name):
			msg = f'Property {prop_name} not defined on socket {self}'
			raise RuntimeError(msg)

		self.trigger_action(ct.DataFlowAction.DataChanged, prop_name=prop_name)

	def trigger_action(
		self,
		action: ct.DataFlowAction,
		socket_name: ct.SocketName | None = None,
		prop_name: ct.SocketName | None = None,
	) -> None:
		"""Reports that the input socket is changed.

		Invalidates (recursively) the cache of any managed object or
		output socket method that implicitly depends on this input socket.
		"""
		# log.debug(
		# 'Action "%s" Triggered in "%s" (socket_name="%s", prop_name="%s")',
		# action,
		# self.name,
		# socket_name,
		# prop_name,
		# )
		# Forwards Chains
		if action == ct.DataFlowAction.DataChanged:
			# Run User Callbacks
			## Careful with these, they run BEFORE propagation...
			## ...because later-chain methods may rely on the results of this.
			for method in self._on_value_changed_methods:
				if (
					(
						socket_name
						and socket_name in method.extra_data['changed_sockets']
					)
					or (prop_name and prop_name in method.extra_data['changed_props'])
					or (
						socket_name
						and method.extra_data['changed_loose_input']
						and socket_name in self.loose_input_sockets
					)
				):
					# log.debug(
					# 'Running Value-Change Callback "%s" in "%s")',
					# method.__name__,
					# self.name,
					# )
					method(self)

			# Propagate via Output Sockets
			for bl_socket in self.active_bl_sockets('output'):
				bl_socket.trigger_action(action)

		# Backwards Chains
		elif action == ct.DataFlowAction.EnableLock:
			self.locked = True

			## Propagate via Input Sockets
			for bl_socket in self.active_bl_sockets('input'):
				bl_socket.trigger_action(action)

		elif action == ct.DataFlowAction.DisableLock:
			self.locked = False

			## Propagate via Input Sockets
			for bl_socket in self.active_bl_sockets('input'):
				bl_socket.trigger_action(action)

		elif action == ct.DataFlowAction.ShowPreview:
			# Run User Callbacks
			## "On Show Preview" callbacks are 'on_value_changed' callbacks...
			## ...which simply hook into the 'preview_active' property.
			## By (maybe) altering 'preview_active', callbacks run as needed.
			if not self.preview_active:
				log.info(
					'Activating Preview in "%s")',
					self.name,
				)
				self.preview_active = True

			## Propagate via Input Sockets
			for bl_socket in self.active_bl_sockets('input'):
				bl_socket.trigger_action(action)

		elif action == ct.DataFlowAction.ShowPlot:
			# Run User Callbacks
			## These shouldn't change any data, BUT...
			## ...because they can stop propagation, they should go first.
			for method in self._on_show_plot:
				method(self)
				if method.extra_data['stop_propagation']:
					return

			## Propagate via Input Sockets
			for bl_socket in self.active_bl_sockets('input'):
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
		CACHE[self.instance_id] = {}

		# Initialize Name
		self.sim_node_name = self.name
		## Only shown in draw_buttons if 'self.use_sim_node_name'

		# Initialize Sockets
		self.sync_sockets()

		# Apply Default Preset
		if self.active_preset:
			self.sync_active_preset()

		# Callbacks
		for method in self._on_init:
			method(self)

	def update(self) -> None:
		pass

	def free(self) -> None:
		"""Run (by Blender) when deleting the node."""
		if not CACHE.get(self.instance_id):
			CACHE[self.instance_id] = {}
		node_tree = self.id_data

		# Unlock
		## This is one approach to the "deleted locked nodes" problem.
		## Essentially, deleting a locked node will unlock along input chain.
		## It also counts if any of the input sockets are linked and locked.
		## Thus, we prevent "dangling locks".
		## TODO: Don't even allow deleting a locked node.
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

		# Finally: Free Instance Cache
		if self.instance_id in CACHE:
			del CACHE[self.instance_id]
