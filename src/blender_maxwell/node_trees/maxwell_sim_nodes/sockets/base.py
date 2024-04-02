import functools
import typing as typ

import bpy
import sympy as sp
import sympy.physics.units as spu
import typing_extensions as typx

from .. import contracts as ct


class MaxwellSimSocket(bpy.types.NodeSocket):
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
	def __init_subclass__(cls, **kwargs: typ.Any):
		super().__init_subclass__(**kwargs)  ## Yucky superclass setup.

		# Setup Blender ID for Node
		if not hasattr(cls, 'socket_type'):
			msg = f"Socket class {cls} does not define 'socket_type'"
			raise ValueError(msg)
		cls.bl_idname = str(cls.socket_type.value)

		# Setup Locked Property for Node
		cls.__annotations__['locked'] = bpy.props.BoolProperty(
			name='Locked State',
			description="The lock-state of a particular socket, which determines the socket's user editability",
			default=False,
		)

		# Setup Style
		cls.socket_color = ct.SOCKET_COLORS[cls.socket_type]
		cls.socket_shape = ct.SOCKET_SHAPES[cls.socket_type]

		# Setup List
		cls.__annotations__['is_list'] = bpy.props.BoolProperty(
			name='Is List',
			description='Whether or not a particular socket is a list type socket',
			default=False,
			update=lambda self, context: self.sync_is_list(context),
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
				update=lambda self, context: self.sync_unit_change(),
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
		action: typx.Literal[
			'enable_lock',
			'disable_lock',
			'value_changed',
			'show_preview',
			'show_plot',
		],
	) -> None:
		"""Called whenever the socket's output value has changed.

		This also invalidates any of the socket's caches.

		When called on an input node, the containing node's
		`trigger_action` method will be called with this socket.

		When called on a linked output node, the linked socket's
		`trigger_action` method will be called.
		"""
		# Forwards Chains
		if action in {'value_changed'}:
			## Input Socket
			if not self.is_output:
				self.node.trigger_action(action, socket_name=self.name)

			## Linked Output Socket
			elif self.is_output and self.is_linked:
				for link in self.links:
					link.to_socket.trigger_action(action)

		# Backwards Chains
		elif action in {
			'enable_lock',
			'disable_lock',
			'show_preview',
			'show_plot',
		}:
			if action == 'enable_lock':
				self.locked = True

			if action == 'disable_lock':
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
	def sync_is_list(self, context: bpy.types.Context):
		"""Called when the "is_list_ property has been updated."""
		if self.is_list:
			if self.use_units:
				self.display_shape = 'SQUARE_DOT'
			else:
				self.display_shape = 'SQUARE'

		self.trigger_action('value_changed')

	def sync_prop(self, prop_name: str, context: bpy.types.Context):
		"""Called when a property has been updated."""
		if not hasattr(self, prop_name):
			msg = f'Property {prop_name} not defined on socket {self}'
			raise RuntimeError(msg)

		self.trigger_action('value_changed')

	def sync_link_added(self, link) -> bool:
		"""Called when a link has been added to this (input) socket.

		Returns a bool, whether or not the socket consents to the link change.
		"""
		if self.locked:
			return False
		if self.is_output:
			msg = "Tried to sync 'link add' on output socket"
			raise RuntimeError(msg)

		self.trigger_action('value_changed')

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

		self.trigger_action('value_changed')

		return True

	####################
	# - Data Chain
	####################
	@property
	def value(self) -> typ.Any:
		raise NotImplementedError

	@value.setter
	def value(self, value: typ.Any) -> None:
		raise NotImplementedError

	@property
	def value_list(self) -> typ.Any:
		return [self.value]

	@value_list.setter
	def value_list(self, value: typ.Any) -> None:
		raise NotImplementedError

	def value_as_unit_system(
		self, unit_system: dict, dimensionless: bool = True
	) -> typ.Any:
		## TODO: Caching could speed this boi up quite a bit

		unit_system_unit = unit_system[self.socket_type]
		return (
			spu.convert_to(
				self.value,
				unit_system_unit,
			)
			/ unit_system_unit
		)

	@property
	def lazy_value(self) -> None:
		raise NotImplementedError

	@lazy_value.setter
	def lazy_value(self, lazy_value: typ.Any) -> None:
		raise NotImplementedError

	@property
	def lazy_value_list(self) -> typ.Any:
		return [self.lazy_value]

	@lazy_value_list.setter
	def lazy_value_list(self, value: typ.Any) -> None:
		raise NotImplementedError

	@property
	def capabilities(self) -> None:
		raise NotImplementedError

	def _compute_data(
		self,
		kind: ct.DataFlowKind = ct.DataFlowKind.Value,
	) -> typ.Any:
		"""Computes the internal data of this socket, ONLY.

		**NOTE**: Low-level method. Use `compute_data` instead.
		"""
		if kind == ct.DataFlowKind.Value:
			if self.is_list:
				return self.value_list
			else:
				return self.value
		elif kind == ct.DataFlowKind.LazyValue:
			if self.is_list:
				return self.lazy_value_list
			else:
				return self.lazy_value
			return self.lazy_value
		elif kind == ct.DataFlowKind.Capabilities:
			return self.capabilities

		return None

	def compute_data(
		self,
		kind: ct.DataFlowKind = ct.DataFlowKind.Value,
	):
		"""Computes the value of this socket, including all relevant factors:
		- If input socket, and unlinked, compute internal data.
		- If input socket, and linked, compute linked socket data.
		- If output socket, ask node for data.
		"""
		# Compute Output Socket
		## List-like sockets guarantee that a list of a thing is passed.
		if self.is_output:
			res = self.node.compute_output(self.name, kind=kind)
			if self.is_list and not isinstance(res, list):
				return [res]
			return res

		# Compute Input Socket
		## Unlinked: Retrieve Socket Value
		if not self.is_linked:
			return self._compute_data(kind)

		## Linked: Compute Output of Linked Sockets
		linked_values = [link.from_socket.compute_data(kind) for link in self.links]

		## Return Single Value / List of Values
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
			msg = f"Tried to set unit for socket {self} with value {value}, but it is not one of possible units {''.join(possible.units.values())} for this socket (as defined in `contracts.SOCKET_UNITS`)"
			raise ValueError(msg)

		if len(matching_unit_names) > 1:
			msg = f"Tried to set unit for socket {self} with value {value}, but multiple possible matching units {''.join(possible.units.values())} for this socket (as defined in `contracts.SOCKET_UNITS`); there may only be one"
			raise RuntimeError(msg)

		self.active_unit = matching_unit_names[0]

	def sync_unit_change(self) -> None:
		"""In unit-aware sockets, the internal `value()` property multiplies the Blender property value by the current active unit.

		When the unit is changed, `value()` will display the old scalar with the new unit.
		To fix this, we need to update the scalar to use the new unit.

		Can be overridden if more specific logic is required.
		"""
		prev_value = self.value / self.unit * self.prev_unit
		## After changing units, self.value is expressed in the wrong unit.
		## - Therefore, we removing the new unit, and re-add the prev unit.
		## - Using only self.value avoids implementation-specific details.

		self.value = spu.convert_to(
			prev_value, self.unit
		)  ## Now, the unit conversion can be done correctly.

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

		# Value Column(s)
		col = row.column(align=True)
		if self.is_list:
			self.draw_value_list(col)
		else:
			self.draw_value(col)

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

	def draw_value(self, col: bpy.types.UILayout) -> None:
		"""Called to draw the value column in unlinked input sockets.

		Can be overridden.
		"""

	def draw_value_list(self, col: bpy.types.UILayout) -> None:
		"""Called to draw the value list column in unlinked input sockets.

		Can be overridden.
		"""
