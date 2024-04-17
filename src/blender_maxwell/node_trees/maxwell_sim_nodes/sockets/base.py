import abc
import functools
import typing as typ

import bpy
import pydantic as pyd
import sympy as sp
import typing_extensions as typx

from ....utils import logger, serialize
from .. import contracts as ct

log = logger.get(__name__)


####################
# - SocketDef
####################
class SocketDef(pyd.BaseModel, abc.ABC):
	socket_type: ct.SocketType

	@abc.abstractmethod
	def init(self, bl_socket: bpy.types.NodeSocket) -> None:
		"""Initializes a real Blender node socket from this socket definition."""

	####################
	# - Serialization
	####################
	def dump_as_msgspec(self) -> serialize.NaiveRepresentation:
		return [serialize.TypeID.SocketDef, self.__class__.__name__, self.model_dump()]

	@staticmethod
	def parse_as_msgspec(obj: serialize.NaiveRepresentation) -> typ.Self:
		return next(
			subclass(**obj[2])
			for subclass in SocketDef.__subclasses__()
			if subclass.__name__ == obj[1]
		)


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
	# link_limit: int = 0
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
		cls.socket_shape = ct.SOCKET_SHAPES[cls.socket_type]

		# Setup List
		cls.__annotations__['active_kind'] = bpy.props.StringProperty(
			name='Active Kind',
			description='The active Data Flow Kind',
			default=str(ct.FlowKind.Value),
			update=lambda self, _: self.sync_active_kind(),
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
	# - Action Chain
	####################
	def trigger_action(
		self,
		action: ct.DataFlowAction,
	) -> None:
		"""Called whenever the socket's output value has changed.

		This also invalidates any of the socket's caches.

		When called on an input node, the containing node's
		`trigger_action` method will be called with this socket.

		When called on a linked output node, the linked socket's
		`trigger_action` method will be called.
		"""
		# Forwards Chains
		if action in {ct.DataFlowAction.DataChanged}:
			## Input Socket
			if not self.is_output:
				self.node.trigger_action(action, socket_name=self.name)

			## Linked Output Socket
			elif self.is_output and self.is_linked:
				for link in self.links:
					link.to_socket.trigger_action(action)

		# Backwards Chains
		elif action in {
			ct.DataFlowAction.EnableLock,
			ct.DataFlowAction.DisableLock,
			ct.DataFlowAction.OutputRequested,
			ct.DataFlowAction.DataChanged,
			ct.DataFlowAction.ShowPreview,
			ct.DataFlowAction.ShowPlot,
		}:
			if action == ct.DataFlowAction.EnableLock:
				self.locked = True

			if action == ct.DataFlowAction.DisableLock:
				self.locked = False

			## Output Socket
			if self.is_output:
				self.node.trigger_action(action, socket_name=self.name)

			## Linked Input Socket
			elif not self.is_output and self.is_linked:
				for link in self.links:
					link.from_socket.trigger_action(action)

	####################
	# - Action Chain: Event Handlers
	####################
	def sync_active_kind(self):
		"""Called when the active data flow kind of the socket changes.

		Alters the shape of the socket to match the active FlowKind, then triggers `ct.DataFlowAction.DataChanged` on the current socket.
		"""
		self.display_shape = {
			ct.FlowKind.Value: ct.SOCKET_SHAPES[self.socket_type],
			ct.FlowKind.ValueArray: 'SQUARE',
			ct.FlowKind.ValueSpectrum: 'SQUARE',
			ct.FlowKind.LazyValue: ct.SOCKET_SHAPES[self.socket_type],
			ct.FlowKind.LazyValueRange: 'SQUARE',
			ct.FlowKind.LazyValueSpectrum: 'SQUARE',
		}[self.active_kind] + ('_DOT' if self.use_units else '')

		self.trigger_action(ct.DataFlowAction.DataChanged)

	def sync_prop(self, prop_name: str, _: bpy.types.Context):
		"""Called when a property has been updated."""
		if hasattr(self, prop_name):
			self.trigger_action(ct.DataFlowAction.DataChanged)
		else:
			msg = f'Property {prop_name} not defined on socket {self}'
			raise RuntimeError(msg)

	def sync_link_added(self, link) -> bool:
		"""Called when a link has been added to this (input) socket."""
		if self.locked:
			log.error(
				'Attempted to link output socket "%s" (%s) to input socket "%s" (%s), but input socket is locked',
				link.from_socket.bl_label,
				link.from_socket.capabilities,
				self.bl_label,
				self.capabilities,
			)
			return False
		if not link.from_socket.capabilities.is_compatible_with(self.capabilities):
			log.error(
				'Attempted to link output socket "%s" (%s) to input socket "%s" (%s), but capabilities are invalid',
				link.from_socket.bl_label,
				link.from_socket.capabilities,
				self.bl_label,
				self.capabilities,
			)
			return False
		if self.is_output:
			msg = "Tried to sync 'link add' on output socket"
			raise RuntimeError(msg)

		self.trigger_action(ct.DataFlowAction.DataChanged)

		return True

	def sync_link_removed(self, from_socket) -> bool:
		"""Called when a link has been removed from this (input) socket.

		Returns a bool, whether or not the socket consents to the link change.
		"""
		if self.locked:
			return False
		if self.is_output:
			msg = "Tried to sync 'link add' on output socket"
			raise RuntimeError(msg)

		self.trigger_action(ct.DataFlowAction.DataChanged)

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
	def value(self) -> ct.DataValue:
		raise NotImplementedError

	@value.setter
	def value(self, value: ct.DataValue) -> None:
		raise NotImplementedError

	# ValueArray
	@property
	def value_array(self) -> ct.DataValueArray:
		raise NotImplementedError

	@value_array.setter
	def value_array(self, value: ct.DataValueArray) -> None:
		raise NotImplementedError

	# ValueSpectrum
	@property
	def value_spectrum(self) -> ct.DataValueSpectrum:
		raise NotImplementedError

	@value_spectrum.setter
	def value_spectrum(self, value: ct.DataValueSpectrum) -> None:
		raise NotImplementedError

	# LazyValue
	@property
	def lazy_value(self) -> ct.LazyDataValue:
		raise NotImplementedError

	@lazy_value.setter
	def lazy_value(self, lazy_value: ct.LazyDataValue) -> None:
		raise NotImplementedError

	# LazyValueRange
	@property
	def lazy_value_range(self) -> ct.LazyDataValueRange:
		raise NotImplementedError

	@lazy_value_range.setter
	def lazy_value_range(self, value: tuple[ct.DataValue, ct.DataValue, int]) -> None:
		raise NotImplementedError

	# LazyValueSpectrum
	@property
	def lazy_value_spectrum(self) -> ct.LazyDataValueSpectrum:
		raise NotImplementedError

	@lazy_value_spectrum.setter
	def lazy_value_spectrum(self, value: ct.LazyDataValueSpectrum) -> None:
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
			ct.FlowKind.ValueSpectrum: lambda: self.value_spectrum,
			ct.FlowKind.LazyValue: lambda: self.lazy_value,
			ct.FlowKind.LazyValueRange: lambda: self.lazy_value_range,
			ct.FlowKind.LazyValueSpectrum: lambda: self.lazy_value_spectrum,
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
