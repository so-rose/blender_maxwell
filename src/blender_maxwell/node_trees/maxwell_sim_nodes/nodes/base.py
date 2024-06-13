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
import functools
import typing as typ
from collections import defaultdict
from types import MappingProxyType

import bpy

from blender_maxwell.utils import bl_cache, bl_instance, logger
from blender_maxwell.utils import sympy_extra as spux

from .. import contracts as ct
from .. import managed_objs as _managed_objs
from .. import sockets
from . import events
from . import presets as _presets

log = logger.get(__name__)

FK = ct.FlowKind
FS = ct.FlowSignal
FE = ct.FlowEvent
MT = spux.MathType
PT = spux.PhysicalType

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

	managed_obj_types: typ.ClassVar[dict[str, ManagedObjs]] = MappingProxyType({})
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
			if hasattr(method := getattr(cls, attr_name), 'identifier')
			and isinstance(method.identifier, str)
			and method.identifier == events.EVENT_METHOD_IDENTIFIER
			## We must not trigger __get__ on any blfields here.
		]
		event_methods_by_event = {event: [] for event in set(FE)}
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
		# log.debug(
		# 'Changed Sim Node Name of a "%s" to "%s" (self=%s)',
		# self.bl_idname,
		# props['sim_node_name'],
		# str(self),
		# )

		# (Re)Construct Managed Objects
		## -> Due to 'prev_name', the new MObjs will be renamed on construction
		managed_objs = props['managed_objs']
		managed_obj_types = props['managed_obj_types']
		self.managed_objs = {
			mobj_name: mobj_type(
				self.sim_node_name + (f'_{i}' if i > 0 else ''),
				prev_name=(
					managed_objs[mobj_name].name if mobj_name in managed_objs else None
				),
			)
			for i, (mobj_name, mobj_type) in enumerate(managed_obj_types.items())
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
	def _prune_inactive_sockets(self):
		"""Remove all inactive sockets from the node, while only updating sockets that can be non-destructively updated.

		The first step is easy: We determine, by-name, which sockets should no longer be defined, then remove them correctly.

		The second step is harder: When new sockets have overlapping names, should they be removed, or should they merely have some properties updated?
		Removing and re-adding the same socket is an accurate, generally robust approach, but it comes with a big caveat: **Existing node links will be cut**, even when it might semantically make sense to simply alter the socket's properties, keeping the links.

		Different `bl_socket.socket_type`s can never be updated - they must be removed.
		Otherwise, `SocketDef.compare(bl_socket)` allows us to granularly determine whether a particular `bl_socket` has changed with respect to the desired specification.
		When the comparison is `False`, we can carefully utilize `SocketDef.init()` to re-initialize the socket, guaranteeing that the altered socket is up to the new specification.
		"""
		node_tree = self.id_data
		for direc in ['input', 'output']:
			active_socket_nametype = {
				bl_socket.name: bl_socket.socket_type
				for bl_socket in self._bl_sockets(direc)
			}
			active_socket_defs = self.active_socket_defs(direc)

			# Determine Sockets to Remove
			## -> Name: If the existing socket name isn't "active".
			## -> Type: If the existing socket_type != "active" SocketDef.
			bl_sockets_to_remove = [
				active_sckname
				for active_sckname, active_scktype in active_socket_nametype.items()
				if (
					active_sckname not in active_socket_defs
					or active_scktype
					is not active_socket_defs[active_sckname].socket_type
				)
			]

			# Determine Sockets to Update
			## -> Name: If the existing socket name is "active".
			## -> Type: If the existing socket_type == "active" SocketDef.
			## -> Compare: If the existing socket differs from the SocketDef.
			## -> NOTE: Reload bl_sockets in case to-update scks were removed.
			bl_sockets_to_update = [
				active_sckname
				for active_sckname, active_scktype in active_socket_nametype.items()
				if (
					active_sckname in active_socket_defs
					and active_scktype is active_socket_defs[active_sckname].socket_type
					and not active_socket_defs[active_sckname].compare(
						self._bl_sockets(direc)[active_sckname]
					)
				)
			]

			# Remove Sockets
			## -> The symptom of using a deleted socket is... hard crash.
			## -> Therefore, we must be EXTREMELY careful with bl_socket refs.
			## -> The multi-stage for-loop helps us guard from deleted sockets.
			for active_sckname in bl_sockets_to_remove:
				bl_socket = self._bl_sockets(direc).get(active_sckname)

				# 1. Report the socket removal to the NodeTree.
				## -> The NodeLinkCache needs to be adjusted manually.
				node_tree.on_node_socket_removed(bl_socket)

			for active_sckname in bl_sockets_to_remove:
				bl_sockets = self._bl_sockets(direc)
				bl_socket = bl_sockets.get(active_sckname)

				# 2. Perform the removal using Blender's API.
				## -> Actually removes the socket.
				## -> Must be protected from auto-removed use-after-free.
				if bl_socket is not None:
					bl_sockets.remove(bl_socket)

				if direc == 'input':
					# 3. Invalidate the input socket cache across all kinds.
					## -> Prevents phantom values from remaining available.
					## -> Done after socket removal to protect from race condition.
					self._compute_input.invalidate(
						input_socket_name=active_sckname,
						kind=...,
						unit_system=...,
					)

					# 4. Run all trigger-only `on_value_changed` callbacks.
					## -> Runs any event methods that relied on the socket.
					## -> Only methods that don't **require** the socket.
					## Only Trigger: If method loads no socket data, it runs.
					## Optional: If method optional-loads socket, it runs.
					triggered_event_methods = [
						value_changed_method
						for value_changed_method in self.event_methods_by_event[
							FE.DataChanged
						]
						if value_changed_method.callback_info.should_run(
							None,
							active_sckname,
							frozenset(FK),
							active_sckname in self.loose_input_sockets,
						)
						and active_sckname
						in value_changed_method.callback_info.optional_sockets_kinds
					]
					for event_method in triggered_event_methods:
						event_method(self)

				else:
					# 3. Invalidate the output socket cache across all kinds.
					## -> Prevents phantom values from remaining available.
					## -> Done after socket removal to protect from race condition.
					self.compute_output.invalidate(
						input_socket_name=active_sckname,
						kind=...,
						unit_system=...,
					)

			# Update Sockets
			## -> The symptom of using a deleted socket is... hard crash.
			## -> Therefore, we must be EXTREMELY careful with bl_socket refs.
			## -> The multi-stage for-loop helps us guard from deleted sockets.
			for active_sckname in bl_sockets_to_update:
				bl_sockets = self._bl_sockets(direc)
				bl_socket = bl_sockets.get(active_sckname)

				if bl_socket is not None:
					socket_def = active_socket_defs[active_sckname]

					# 1. Pretend to Initialize for the First Time
					## -> NOTE: The socket's caches will be completely regenerated.
					## -> NOTE: A full FlowKind update will occur, but only one.
					bl_socket.is_initializing = True
					socket_def.preinit(bl_socket)
					socket_def.init(bl_socket)
					socket_def.postinit(bl_socket)

			for active_sckname in bl_sockets_to_update:
				bl_sockets = self._bl_sockets(direc)
				bl_socket = bl_sockets.get(active_sckname)

				if bl_socket is not None:
					# 2. Re-Test Socket Capabilities
					## -> Factors influencing CapabilitiesFlow may have changed.
					## -> Therefore, we must re-test all link capabilities.
					bl_socket.remove_invalidated_links()

					if direc == 'input':
						# 3. Invalidate the input socket cache across all kinds.
						## -> Prevents phantom values from remaining available.
						self._compute_input.invalidate(
							input_socket_name=active_sckname,
							kind=...,
							unit_system=...,
						)

					if direc == 'output':
						# 3. Invalidate the output socket cache across all kinds.
						## -> Prevents phantom values from remaining available.
						## -> Done after socket removal to protect from race condition.
						self.compute_output.invalidate(
							input_socket_name=active_sckname,
							kind=...,
							unit_system=...,
						)

	def _add_new_active_sockets(self):
		"""Add and initialize all "active" sockets that aren't on the node.

		Existing sockets within the given direction are not re-created.
		"""
		for direc in ['input', 'output']:
			bl_sockets = self._bl_sockets(direc)
			active_socket_defs = self.active_socket_defs(direc)

			# Define BL Sockets
			created_sockets = {}
			for socket_name, socket_def in active_socket_defs.items():
				# Skip Existing Sockets
				if socket_name in bl_sockets:
					continue

				# Create BL Socket from Socket
				bl_sockets.new(
					str(socket_def.socket_type.value),
					socket_name,
				)

				# Record Socket Creation
				created_sockets[socket_name] = socket_def

			# Initialize Just-Created BL Sockets
			for bl_socket_name, socket_def in created_sockets.items():
				socket_def.preinit(bl_sockets[bl_socket_name])
				socket_def.init(bl_sockets[bl_socket_name])
				socket_def.postinit(bl_sockets[bl_socket_name])

				# Invalidate Cached NoFlows
				self._compute_input.invalidate(
					input_socket_name=bl_socket_name,
					kind=...,
					unit_system=...,
				)

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

		for i, socket_name in enumerate(self.active_socket_defs('input')):
			current_idx_of_socket = self.inputs.find(socket_name)
			if i != current_idx_of_socket:
				self.inputs.move(current_idx_of_socket, i)

		for i, socket_name in enumerate(self.active_socket_defs('output')):
			current_idx_of_socket = self.outputs.find(socket_name)
			if i != current_idx_of_socket:
				self.outputs.move(current_idx_of_socket, i)

	####################
	# - Compute: Input Socket
	####################
	@bl_cache.keyed_cache(
		exclude={'self'},
	)
	def compute_prop(
		self,
		prop_name: ct.PropName,
		unit_system: spux.UnitSystem | None = None,
	) -> typ.Any:
		"""Computes the data of a property, with relevant unit system scaling.

		Some properties return a `sympy` expression, which needs to be conformed to some unit system before it can be used.
		For these cases,

		When no unit system is in use, the cache of `compute_prop` is a transparent layer on top of the `BLField` cache, taking no extra memory.

		Warnings:
			**MUST** be invalidated whenever a property changed.
			If not, then "phantom" values will be produced.

		Parameters:
			prop_name: The name of the property to compute the value of.
			unit_system: The unit system to convert it to, if any.

		Returns:
			The property value, possibly scaled to the unit system.
		"""
		# Retrieve Unit System and Property
		if hasattr(self, prop_name):
			prop_value = getattr(self, prop_name)
		else:
			msg = f'The node {self.sim_node_name} has no property {prop_name}.'
			raise ValueError

		if unit_system is not None:
			if isinstance(prop_value, spux.SympyType):
				return spux.scale_to_unit_system(prop_value)

			msg = f'Cannot scale property {prop_name}={prop_value} (type={type(prop_value)} to a unit system, since it is not a sympy object (unit_system={unit_system})'
			raise ValueError(msg)

		return prop_value

	@bl_cache.keyed_cache(
		exclude={'self'},
	)
	def _compute_input(
		self,
		input_socket_name: ct.SocketName,
		kind: FK = FK.Value,
		unit_system: spux.UnitSystem | None = None,
	) -> typ.Any:
		"""Computes the data of an input socket, following links if needed.

		Notes:
			The semantics derive entirely from `sockets.MaxwellSimSocket.compute_data()`.

		Parameters:
			input_socket_name: The name of the input socket to compute the value of.
				It must be currently active.
			kind: The data flow kind to compute.
			unit_system: The unit system to scale the computed input socket to.
		"""
		bl_socket = self.inputs.get(input_socket_name)
		if bl_socket is not None:
			if bl_socket.instance_id:
				if kind is FK.Previews:
					return bl_socket.compute_data(kind=kind)

				return (
					FK.scale_to_unit_system(
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
			return FS.FlowInitializing

		if kind is FK.Previews:
			return ct.PreviewsFlow()
		return FS.NoFlow

	####################
	# - Compute Event: Output Socket
	####################
	@bl_cache.keyed_cache(
		exclude={'self'},
	)
	def compute_output(
		self,
		output_socket_name: ct.SocketName,
		kind: FK = FK.Value,
		unit_system: spux.UnitSystem | None = None,
	) -> typ.Any | FS:
		"""Computes the value of an output socket.

		Parameters:
			output_socket_name: The name declaring the output socket, for which this method computes the output.
			kind: The FlowKind to use when computing the output socket value.
			unit_system: The unit system to scale the computed output socket to.

		Returns:
			The value of the output socket, as computed by the dedicated method
			registered using the `@computes_output_socket` decorator.
		"""
		log.debug(
			'[%s] Computing Output (socket_name=%s, socket_kinds=%s, unit_system=%s)',
			self.sim_node_name,
			str(output_socket_name),
			str(kind),
			str(unit_system),
		)

		bl_socket = self.outputs.get(output_socket_name)
		if bl_socket is not None:
			if bl_socket.instance_id:
				# Previews: Computed Aggregated Input Sockets
				## -> All sockets w/output get Previews from all inputs.
				## -> The user can also assign certain sockets.
				if kind is FK.Previews:
					input_previews = functools.reduce(
						lambda a, b: a | b,
						[
							self._compute_input(
								socket_name,
								kind=FK.Previews,
							)
							for socket_name in [
								bl_socket.name for bl_socket in self.inputs
							]
						],
						ct.PreviewsFlow(),
					)

				# Retrieve Valid Output Socket Method
				## -> We presume that there is exactly one method per socket|kind.
				## -> We presume that there is exactly one method per socket|kind.
				outsck_methods = [
					method
					for method in self.event_methods_by_event[FE.OutputRequested]
					if method.callback_info.should_run(output_socket_name, kind)
				]
				if len(outsck_methods) != 1:
					if kind is FK.Previews:
						return input_previews
					return FS.NoFlow

				outsck_method = outsck_methods[0]

				# Compute Flow w/Output Socket Method
				flow = outsck_method(self)
				has_flow = not FS.check(flow)

				if kind is FK.Previews:
					if has_flow:
						return input_previews | flow
					return input_previews

				# *: Compute Flow
				## -> Perform unit-system scaling (maybe)
				## -> Otherwise, return flow (even if FlowSignal).
				if has_flow and unit_system is not None:
					return kind.scale_to_unit_system(flow, unit_system)
				return flow

			# No Socket Instance ID
			## -> Indicates that socket_def.preinit() has not yet run.
			## -> Anyone needing results will need to wait on preinit().
			return FS.FlowInitializing
		return FS.NoFlow

	####################
	# - Plot
	####################
	def compute_plot(self):
		"""Run all `on_show_plot` event methods."""
		plot_methods = self.event_methods_by_event[FE.ShowPlot]
		for plot_method in plot_methods:
			plot_method(self)

	####################
	# - Event Trigger
	####################
	def _should_recompute_output_socket(
		self,
		method_info: events.InfoOutputRequested,
		input_socket_name: ct.SocketName | None,
		input_socket_kinds: set[FK] | None,
		prop_names: set[ct.PropName] | None,
	) -> bool:
		return (
			prop_names is not None
			and any(prop_name in method_info.depon_props for prop_name in prop_names)
			or input_socket_name is not None
			and (
				input_socket_name in method_info.depon_input_sockets
				and (
					input_socket_kinds is None
					or (
						isinstance(
							_kind := method_info.depon_input_sockets_kinds.get(
								input_socket_name, FK.Value
							),
							set,
						)
						and input_socket_kinds.intersection(_kind)
					)
					or _kind == FK.Value
					or _kind in input_socket_kinds
				)
				or (
					method_info.depon_all_loose_input_sockets
					and input_socket_name in self.loose_input_sockets
				)
			)
		)

	@bl_cache.cached_bl_property()
	def output_socket_invalidates(
		self,
	) -> dict[tuple[ct.SocketName, FK], set[tuple[ct.SocketName, FK]]]:
		"""Deduce which output socket | `FlowKind` combos are altered in response to a given output socket | `FlowKind` combo.

		Returns:
			A dictionary, wher eeach key is a tuple representing an output socket name and its flow kind that has been altered, and each value is a set of tuples representing output socket names and flow kind.

			Indexing by any particular `(output_socket_name, flow_kind)` will produce a set of all `{(output_socket_name, flow_kind)}` that rely on it.
		"""
		altered_to_invalidated = defaultdict(set)

		# Iterate ALL Methods that Compute Output Sockets
		## -> We call it the "altered method".
		## -> Our approach will be to deduce what relies on it.
		output_requested_methods = self.event_methods_by_event[FE.OutputRequested]
		for altered_method in output_requested_methods:
			altered_info = altered_method.callback_info
			altered_key = (altered_info.output_socket_name, altered_info.kind)

			# Inner: Iterate ALL Methods that Compute Output Sockets
			## -> We call it the "invalidated method".
			## -> While O(n^2), it runs only once per-node, and is then cached.
			## -> `n` is rarely so large as to be a startup-time concern.
			## -> Thus, in this case, using a simple implementation is better.
			for invalidated_method in output_requested_methods:
				invalidated_info = invalidated_method.callback_info

				# Check #0: Inv. Socket depends on Altered Socket
				## -> Simply check if the altered name is in the dependencies.
				if (
					altered_info.output_socket_name
					in invalidated_info.depon_output_sockets
				):
					# Check #2: FlowKinds Match
					## -> Case 1: Single Altered Kind was Requested by Inv
					## -> Case 2: Altered Kind in set[Requested Kinds] is
					## -> Case 3: Altered Kind is FlowKind.Value
					## This encapsulates the actual events decorator semantics.
					is_same_kind = (
						altered_info.kind
						is (
							_kind := invalidated_info.depon_output_sockets_kinds.get(
								altered_info.output_socket_name
							)
						)
						or (
							isinstance(_kind, set | frozenset)
							and altered_info.kind in _kind
						)
						or altered_info.kind is FK.Value
					)

					# Check Success: Add Invalidated (name,kind) to Altered Set
					## -> We've now confirmed a dependency.
					## -> Thus, this name|kind should be included.
					if is_same_kind:
						invalidated_key = (
							invalidated_info.output_socket_name,
							invalidated_info.kind,
						)
						altered_to_invalidated[altered_key].add(invalidated_key)

		return altered_to_invalidated

	def trigger_event(
		self,
		event: FE,
		socket_name: ct.SocketName | None = None,
		socket_kinds: set[FK] | None = None,
		prop_names: set[ct.PropName] | None = None,
	) -> None:
		"""Recursively triggers events forwards or backwards along the node tree, allowing nodes in the update path to react.

		Use `events` decorators to define methods that react to particular `FE`s.

		Notes:
			This can be an unpredictably heavy function, depending on the node graph topology.

			Doesn't accept `LinkChanged` events; they are translated to `DataChanged` on the socket.
			This is on purpose: It seems to be a bad idea to try and differentiate between "changes in data" and "changes in linkage".

		Parameters:
			event: The event to report forwards/backwards along the node tree.
			socket_name: The input socket that was altered, if any, in order to trigger this event.
			pop_name: The property that was altered, if any, in order to trigger this event.
		"""
		if socket_kinds is not None:
			socket_kinds = frozenset(socket_kinds)
		if prop_names is not None:
			prop_names = frozenset(prop_names)

		## -> Track actual alterations per output socket|kind.
		altered_outscks_kinds: dict[ct.SocketName, set[FK]] = defaultdict(set)

		# log.debug(
		# '[%s] [%s] Triggered (socket_name=%s, socket_kinds=%s, prop_names=%s)',
		# self.sim_node_name,
		# event,
		# str(socket_name),
		# str(socket_kinds),
		# str(prop_names),
		# )

		# Event: DataChanged
		if event is FE.DataChanged:
			in_sckname = socket_name

			# Input Socket: Clear Cache(s)
			## -> The input socket cache for each altered FlowKinds is cleared.
			## -> Since it's non-persistent, it will be lazily re-filled.
			if in_sckname is not None:
				for in_kind in socket_kinds:
					# log.debug(
					# '![%s] Clear Input Socket Cache (%s, %s)',
					# self.sim_node_name,
					# in_sckname,
					# in_kind,
					# )
					self._compute_input.invalidate(
						input_socket_name=in_sckname,
						kind=in_kind,
						unit_system=...,
					)

			# Output Sockets: Clear Cache(s)
			for output_socket_method in self.event_methods_by_event[FE.OutputRequested]:
				# Determine Consequences of Changed (Socket|Kind) / Prop
				## -> Each '@computes_output_socket' declares data to load.
				## -> Ask if the altered data should cause it to reload.
				method_info = output_socket_method.callback_info
				if method_info.should_recompute(
					prop_names,
					socket_name,
					socket_kinds,
					socket_name in self.loose_input_sockets,
				):
					out_sckname = method_info.output_socket_name
					out_kind = method_info.kind

					self.compute_output.invalidate(
						output_socket_name=out_sckname,
						kind=out_kind,
						unit_system=...,
					)
					altered_outscks_kinds[out_sckname].add(out_kind)

					# Recurse: Output-Output Dependencies
					## -> Outscks may depend on each other.
					## -> Pre-build an invalidation map per-node.
					cleared_outscks_kinds = self.output_socket_invalidates.get(
						(out_sckname, out_kind)
					)
					if cleared_outscks_kinds:
						for dep_out_sckname, dep_out_kind in cleared_outscks_kinds:
							self.compute_output.invalidate(
								output_socket_name=dep_out_sckname,
								kind=dep_out_kind,
								unit_system=...,
							)
							altered_outscks_kinds[dep_out_sckname].add(dep_out_kind)

			# Any Preview Change -> All Output Previews Regenerate
			## -> All output sockets aggregate all input socket previews.
			if socket_kinds is not None and FK.Previews in socket_kinds:
				for out_sckname in self.outputs.keys():  # noqa: SIM118
					self.compute_output.invalidate(
						output_socket_name=out_sckname,
						kind=FK.Previews,
						unit_system=...,
					)
					altered_outscks_kinds[out_sckname].add(FK.Previews)

			# Run 'on_value_changed' Callbacks
			## -> These event methods specifically respond to DataChanged.
			stop_propagation = False
			for value_changed_method in (
				method
				for method in self.event_methods_by_event[FE.DataChanged]
				if method.callback_info.should_run(
					prop_names,
					socket_name,
					socket_kinds,
					socket_name in self.loose_input_sockets,
				)
			):
				stop_propagation |= value_changed_method.callback_info.stop_propagation
				value_changed_method(self)

			if stop_propagation:
				return

		elif event is FE.EnableLock or event is FE.DisableLock:
			for lock_method in self.event_methods_by_event[event]:
				lock_method(self)

		# Propagate Event
		## -> If 'stop_propagation' was tripped, don't propagate.
		## -> If no sockets were altered during DataChanged, don't propagate.
		## -> Each FlowEvent decides whether to flow forwards/backwards.
		## -> The trigger chain goes node/socket/socket/node/socket/...
		## -> Unlinked sockets naturally stop the propagation.
		direc = FE.flow_direction[event]
		for bl_socket in self._bl_sockets(direc=direc):
			# DataChanged: Propagate Altered SocketKinds
			## -> Only altered FlowKinds for the socket will propagate.
			## -> In this way, we guarantee no extraneous (noop) flow.
			if event is FE.DataChanged:
				if bl_socket.name in altered_outscks_kinds:
					# log.debug(
					# '![%s] [%s] Propagating (direction=%s, altered_socket_kinds=%s)',
					# self.sim_node_name,
					# event,
					# direc,
					# altered_socket_kinds[bl_socket.name],
					# )
					bl_socket.trigger_event(
						event, socket_kinds=altered_outscks_kinds[bl_socket.name]
					)

				## -> Otherwise, do nothing - guarantee no extraneous flow.

			# Propagate Normally
			else:
				# log.debug(
				# '![%s] [%s] Propagating (direction=%s)',
				# self.sim_node_name,
				# event,
				# direc,
				# )
				bl_socket.trigger_event(event)

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
		# BLField Attributes: Invalidate BLField Dependents
		## -> All invalidated blfields will have their caches cleared.
		## -> The (topologically) ordered list of cleared blfields is returned.
		## -> WARNING: The chain is not checked for ex. cycles.
		if prop_name in self.blfields:
			cleared_blfields = self.clear_blfields_after(prop_name)

			for prop_name, _ in cleared_blfields:
				self.compute_prop.invalidate(
					prop_name=prop_name,
					unit_system=...,
				)

			# log.debug(
			# '%s (Node): Set of Cleared BLFields: %s',
			# self.bl_label,
			# str(cleared_blfields),
			# )
			self.trigger_event(
				FE.DataChanged,
				prop_names={prop_name for prop_name, _ in cleared_blfields},
			)

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
			for event_method in self.event_methods_by_event[FE.DataChanged]
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
			for event_method in self.event_methods_by_event[FE.DataChanged]
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
			self.trigger_event(FE.DisableLock)

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
