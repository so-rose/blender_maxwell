import abc
import functools
import typing as typ

import bpy
import pydantic as pyd
import sympy as sp
import typing_extensions as typx

from blender_maxwell.utils import logger, serialize
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
		return next(initialized_classes)


####################
# - SocketDef
####################
class MaxwellSimSocket(bpy.types.NodeSocket):
	"""A specialized Blender socket for nodes in a Maxwell simulation.

	Attributes:
		instance_id: A unique ID attached to a particular socket instance.
			Guaranteed to be unchanged so long as the socket lives.
			Used as a socket-specific cache index.
		locked: The lock-state of a particular socket, which determines the socket's user editability
	"""

	# Fundamentals
	socket_type: ct.SocketType
	bl_label: str

	# Style
	display_shape: typx.Literal[
		'CIRCLE',
		'SQUARE',
		'DIAMOND',
		'CIRCLE_DOT',
		'SQUARE_DOT',
		'DIAMOND_DOT',
	]
	## We use the following conventions for shapes:
	## - CIRCLE: Single Value.
	## - SQUARE: Container of Value.
	## - DIAMOND: Pointer Value.
	## - +DOT: Uses Units
	socket_color: tuple

	# Options
	use_units: bool = False
	use_prelock: bool = False

	# Computed
	bl_idname: str

	####################
	# - Initialization
	####################
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

	def __init_subclass__(cls, **kwargs: typ.Any):
		log.debug('Initializing Socket: %s', cls.socket_type)
		super().__init_subclass__(**kwargs)
		# cls._assert_attrs_valid()

		# Socket Properties
		## Identifiers
		cls.bl_idname: str = str(cls.socket_type.value)
		cls.set_prop('instance_id', bpy.props.StringProperty, no_update=True)

		## Special States
		cls.set_prop('locked', bpy.props.BoolProperty, no_update=True, default=False)

		# Setup Style
		cls.socket_color = ct.SOCKET_COLORS[cls.socket_type]

		# Setup List
		cls.set_prop(
			'active_kind', bpy.props.StringProperty, default=str(ct.FlowKind.Value)
		)

		# Configure Use of Units
		if cls.use_units:
			# Set Shape :)
			cls.socket_shape += '_DOT'

			if not (socket_units := ct.SOCKET_UNITS.get(cls.socket_type)):
				msg = 'Tried to `use_units` on {cls.bl_idname} socket, but `SocketType` has no units defined in `contracts.SOCKET_UNITS`'
				raise RuntimeError(msg)

			# Current Unit
			cls.__annotations__['active_unit'] = bpy.props.EnumProperty(
				name='Unit',
				description='Choose a unit',
				items=[
					(unit_name, str(unit_value), str(unit_value))
					for unit_name, unit_value in socket_units['values'].items()
				],
				default=socket_units['default'],
				update=lambda self, _: self.sync_unit_change(),
			)

			# Previous Unit (for conversion)
			cls.__annotations__['prev_active_unit'] = bpy.props.StringProperty(
				default=socket_units['default'],
			)

	####################
	# - Event Chain
	####################
	def trigger_event(
		self,
		event: ct.FlowEvent,
	) -> None:
		"""Called whenever the socket's output value has changed.

		This also invalidates any of the socket's caches.

		When called on an input node, the containing node's
		`trigger_event` method will be called with this socket.

		When called on a linked output node, the linked socket's
		`trigger_event` method will be called.
		"""
		# Forwards Chains
		if event in {ct.FlowEvent.DataChanged}:
			## Input Socket
			if not self.is_output:
				self.node.trigger_event(event, socket_name=self.name)

			## Linked Output Socket
			elif self.is_output and self.is_linked:
				for link in self.links:
					link.to_socket.trigger_event(event)

		# Backwards Chains
		elif event in {
			ct.FlowEvent.EnableLock,
			ct.FlowEvent.DisableLock,
			ct.FlowEvent.OutputRequested,
			ct.FlowEvent.DataChanged,
			ct.FlowEvent.ShowPreview,
			ct.FlowEvent.ShowPlot,
		}:
			if event == ct.FlowEvent.EnableLock:
				self.locked = True

			if event == ct.FlowEvent.DisableLock:
				self.locked = False

			## Output Socket
			if self.is_output:
				self.node.trigger_event(event, socket_name=self.name)

			## Linked Input Socket
			elif not self.is_output and self.is_linked:
				for link in self.links:
					link.from_socket.trigger_event(event)

	####################
	# - Event Chain: Event Handlers
	####################
	def sync_prop(self, prop_name: str, _: bpy.types.Context) -> None:
		"""Called when a property has been updated.

		Contrary to `node.on_prop_changed()`, socket-specific callbacks are baked into this function:

		- **Active Kind** (`active_kind`): Sets the socket shape to reflect the active `FlowKind`.

		Attributes:
			prop_name: The name of the property that was changed.
		"""
		# Property: Active Kind
		if prop_name == 'active_kind':
			self.display_shape(
				'SQUARE'
				if self.active_kind
				in {ct.FlowKind.LazyValue, ct.FlowKind.LazyValueRange}
				else 'CIRCLE'
			) + ('_DOT' if self.use_units else '')

		# Valid Properties
		elif hasattr(self, prop_name):
			self.trigger_event(ct.FlowEvent.DataChanged)

		# Undefined Properties
		else:
			msg = f'Property {prop_name} not defined on socket {self}'
			raise RuntimeError(msg)

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
			msg = 'Tried to ask output socket for consent to add link'
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

	def on_link_added(self, link: bpy.types.NodeLink) -> None:
		"""Triggers a `ct.FlowEvent.LinkChanged` event on link add.

		Attributes:
			link: The node link that was added.
				Currently unused.

		Returns:
			Whether or not consent is given to add the link.
			In practice, the link will simply remain if consent is given.
			If consent is not given, the new link will be removed.
		"""
		self.trigger_event(ct.FlowEvent.DataChanged)

	def allow_remove_link(self, from_socket: bpy.types.NodeSocket) -> bool:
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
			msg = "Tried to sync 'link add' on output socket"
			raise RuntimeError(msg)

		# Lock Check
		if self.locked:
			return False

		self.trigger_event(ct.FlowEvent.DataChanged)
		return True

	####################
	# - Data Chain
	####################
	# Capabilities
	@property
	def capabilities(self) -> None:
		return ct.DataCapabilities(
			socket_type=self.socket_type,
			active_kind=self.active_kind,
		)

	# Value
	@property
	def value(self) -> ct.ValueFlow:
		raise NotImplementedError

	@value.setter
	def value(self, value: ct.ValueFlow) -> None:
		raise NotImplementedError

	# ValueArray
	@property
	def array(self) -> ct.ArrayFlow:
		## TODO: Single-element list when value exists.
		raise NotImplementedError

	@array.setter
	def array(self, value: ct.ArrayFlow) -> None:
		raise NotImplementedError

	# LazyValue
	@property
	def lazy_value(self) -> ct.LazyValueFlow:
		raise NotImplementedError

	@lazy_value.setter
	def lazy_value(self, lazy_value: ct.LazyValueFlow) -> None:
		raise NotImplementedError

	# LazyArrayRange
	@property
	def lazy_array_range(self) -> ct.LazyArrayRangeFlow:
		raise NotImplementedError

	@lazy_array_range.setter
	def lazy_array_range(self, value: tuple[ct.DataValue, ct.DataValue, int]) -> None:
		raise NotImplementedError

	# LazyArrayRange
	@property
	def param(self) -> ct.ParamsFlow:
		raise NotImplementedError

	@param.setter
	def param(self, value: tuple[ct.DataValue, ct.DataValue, int]) -> None:
		raise NotImplementedError

	####################
	# - Data Chain Computation
	####################
	def _compute_data(
		self,
		kind: ct.FlowKind = ct.FlowKind.Value,
	) -> typ.Any:
		"""Computes the internal data of this socket, ONLY.

		**NOTE**: Low-level method. Use `compute_data` instead.
		"""
		return {
			ct.FlowKind.Value: lambda: self.value,
			ct.FlowKind.ValueArray: lambda: self.value_array,
			ct.FlowKind.LazyValue: lambda: self.lazy_value,
			ct.FlowKind.LazyArrayRange: lambda: self.lazy_array_range,
			ct.FlowKind.Params: lambda: self.params,
			ct.FlowKind.Info: lambda: self.info,
		}[kind]()

		msg = f'socket._compute_data was called with invalid kind "{kind}"'
		raise RuntimeError(msg)

	def compute_data(
		self,
		kind: ct.FlowKind = ct.FlowKind.Value,
	):
		"""Computes the value of this socket, including all relevant factors.

		Notes:
			- If input socket, and unlinked, compute internal data.
			- If input socket, and linked, compute linked socket data.
			- If output socket, ask node for data.
		"""
		# Compute Output Socket
		if self.is_output:
			return self.node.compute_output(self.name, kind=kind)

		# Compute Input Socket
		## Unlinked: Retrieve Socket Value
		if not self.is_linked:
			return self._compute_data(kind)

		## Linked: Check Capabilities
		for link in self.links:
			if not link.from_socket.capabilities.is_compatible_with(self.capabilities):
				msg = f'Output socket "{link.from_socket.bl_label}" is linked to input socket "{self.bl_label}" with incompatible capabilities (caps_out="{link.from_socket.capabilities}", caps_in="{self.capabilities}")'
				raise ValueError(msg)

		## ...and Compute Data on Linked Socket
		linked_values = [link.from_socket.compute_data(kind) for link in self.links]

		# Return Single Value / List of Values
		## Preparation for multi-input sockets.
		if len(linked_values) == 1:
			return linked_values[0]
		return linked_values

	####################
	# - Unit Properties
	####################
	@functools.cached_property
	def possible_units(self) -> dict[str, sp.Expr]:
		if not self.use_units:
			msg = "Tried to get possible units for socket {self}, but socket doesn't `use_units`"
			raise ValueError(msg)

		return ct.SOCKET_UNITS[self.socket_type]['values']

	@property
	def unit(self) -> sp.Expr:
		return self.possible_units[self.active_unit]

	@property
	def prev_unit(self) -> sp.Expr:
		return self.possible_units[self.prev_active_unit]

	@unit.setter
	def unit(self, value: str | sp.Expr) -> None:
		# Retrieve Unit by String
		if isinstance(value, str) and value in self.possible_units:
			self.active_unit = self.possible_units[value]
			return

		# Retrieve =1 Matching Unit Name
		matching_unit_names = [
			unit_name
			for unit_name, unit_sympy in self.possible_units.items()
			if value == unit_sympy
		]
		if len(matching_unit_names) == 0:
			msg = f"Tried to set unit for socket {self} with value {value}, but it is not one of possible units {''.join(self.possible_units.values())} for this socket (as defined in `contracts.SOCKET_UNITS`)"
			raise ValueError(msg)

		if len(matching_unit_names) > 1:
			msg = f"Tried to set unit for socket {self} with value {value}, but multiple possible matching units {''.join(self.possible_units.values())} for this socket (as defined in `contracts.SOCKET_UNITS`); there may only be one"
			raise RuntimeError(msg)

		self.active_unit = matching_unit_names[0]

	def sync_unit_change(self) -> None:
		"""In unit-aware sockets, the internal `value()` property multiplies the Blender property value by the current active unit.

		When the unit is changed, `value()` will display the old scalar with the new unit.
		To fix this, we need to update the scalar to use the new unit.

		Can be overridden if more specific logic is required.
		"""
		if self.active_kind == ct.FlowKind.Value:
			self.value = self.value / self.unit * self.prev_unit

		elif self.active_kind == ct.FlowKind.LazyValueRange:
			lazy_value_range = self.lazy_value_range
			self.lazy_value_range = (
				lazy_value_range.start / self.unit * self.prev_unit,
				lazy_value_range.stop / self.unit * self.prev_unit,
				lazy_value_range.steps,
			)

		self.prev_active_unit = self.active_unit

	####################
	# - Style
	####################
	def draw_color(
		self,
		context: bpy.types.Context,
		node: bpy.types.Node,
	) -> ct.BLColorRGBA:
		"""Color of the socket icon, when embedded in a node."""
		return self.socket_color

	@classmethod
	def draw_color_simple(cls) -> ct.BLColorRGBA:
		"""Fallback color of the socket icon (ex.when not embedded in a node)."""
		return cls.socket_color

	####################
	# - UI Methods
	####################
	def draw(
		self,
		context: bpy.types.Context,
		layout: bpy.types.UILayout,
		node: bpy.types.Node,
		text: str,
	) -> None:
		"""Called by Blender to draw the socket UI."""
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
		pass

	def draw_input(
		self,
		context: bpy.types.Context,
		layout: bpy.types.UILayout,
		node: bpy.types.Node,
		text: str,
	) -> None:
		"""Draws the socket UI, when the socket is an input socket."""
		col = layout.column(align=False)

		# Label Row
		row = col.row(align=False)
		if self.locked:
			row.enabled = False

		## Linked Label
		if self.is_linked:
			row.label(text=text)
			return

		## User Label Row (incl. Units)
		if self.use_units:
			split = row.split(factor=0.6, align=True)

			_row = split.row(align=True)
			self.draw_label_row(_row, text)

			_col = split.column(align=True)
			_col.prop(self, 'active_unit', text='')
		else:
			self.draw_label_row(row, text)

		# Prelock Row
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

		# Data Column(s)
		col = row.column(align=True)
		{
			ct.FlowKind.Value: self.draw_value,
			ct.FlowKind.ValueArray: self.draw_value_array,
			ct.FlowKind.ValueSpectrum: self.draw_value_spectrum,
			ct.FlowKind.LazyValue: self.draw_lazy_value,
			ct.FlowKind.LazyValueRange: self.draw_lazy_value_range,
			ct.FlowKind.LazyValueSpectrum: self.draw_lazy_value_spectrum,
		}[self.active_kind](col)

	def draw_output(
		self,
		context: bpy.types.Context,
		layout: bpy.types.UILayout,
		node: bpy.types.Node,
		text: str,
	) -> None:
		"""Draws the socket UI, when the socket is an output socket."""
		layout.label(text=text)

	####################
	# - UI Methods
	####################
	def draw_label_row(
		self,
		row: bpy.types.UILayout,
		text: str,
	) -> None:
		"""Called to draw the label row (same height as socket shape).

		Can be overridden.
		"""
		row.label(text=text)

	####################
	# - FlowKind draw() Methods
	####################
	def draw_value(self, col: bpy.types.UILayout) -> None:
		pass

	def draw_value_array(self, col: bpy.types.UILayout) -> None:
		pass

	def draw_value_spectrum(self, col: bpy.types.UILayout) -> None:
		pass

	def draw_lazy_value(self, col: bpy.types.UILayout) -> None:
		pass

	def draw_lazy_value_range(self, col: bpy.types.UILayout) -> None:
		pass

	def draw_lazy_value_spectrum(self, col: bpy.types.UILayout) -> None:
		pass
