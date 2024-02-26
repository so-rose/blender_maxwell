import typing as typ
import bpy

import sympy as sp
import sympy.physics.units as spu
from .. import contracts

class BLSocket(bpy.types.NodeSocket):
	"""A base type for nodes that greatly simplifies the implementation of
	reliable, powerful nodes.
	
	Should be used together with `contracts.BLSocketProtocol`.
	"""
	def __init_subclass__(cls, **kwargs: typ.Any):
		super().__init_subclass__(**kwargs)  ## Yucky superclass setup.
		
		# Set bl_idname
		cls.bl_idname = cls.socket_type.value 
		cls.socket_color = contracts.SocketType_to_color[
			cls.socket_type.value
		]
		
		# Configure Use of Units
		if (
			hasattr(cls, "use_units")
			and cls.socket_type in contracts.SocketType_to_units
		):
			# Set Unit Properties
			cls.__annotations__["raw_unit"] = bpy.props.EnumProperty(
				name="Unit",
				description="Choose a unit",
				items=[
					(unit_name, str(unit_value), str(unit_value))
					for unit_name, unit_value in contracts.SocketType_to_units[
						cls.socket_type
					]["values"].items()
				],
				default=contracts.SocketType_to_units[
					cls.socket_type
				]["default"],
				update=lambda self, context: self._update_unit(),
			)
			cls.__annotations__["raw_unit_previous"] = bpy.props.StringProperty(
				default=contracts.SocketType_to_units[
					cls.socket_type
				]["default"]
			)
		
		# Declare Node Property: 'preset' EnumProperty
		if hasattr(cls, "draw_preview"):
			cls.__annotations__["preview_active"] = bpy.props.BoolProperty(
				name="Preview",
				description="Preview the socket value",
				default=False,
			)
	
	####################
	# - Internal Methods
	####################
	@property
	def units(self) -> dict[str, sp.Expr]:
		return contracts.SocketType_to_units[
			self.socket_type
		]["values"]
	
	@property
	def unit(self) -> sp.Expr:
		return contracts.SocketType_to_units[
			self.socket_type
		]["values"][self.raw_unit]
	
	@unit.setter
	def unit(self, value) -> sp.Expr:
		raw_unit_name = [
			raw_unit_name
			for raw_unit_name, unit_value in contracts.SocketType_to_units[
				self.socket_type
			]["values"].items()
			if value == unit_value
		][0]
		
		self.raw_unit = raw_unit_name
	
	@property
	def _unit_previous(self) -> sp.Expr:
		return contracts.SocketType_to_units[
			self.socket_type
		]["values"][self.raw_unit_previous]
	
	@_unit_previous.setter
	def _unit_previous(self, value) -> sp.Expr:
		raw_unit_name = [
			raw_unit_name
			for raw_unit_name, unit_value in contracts.SocketType_to_units[
				self.socket_type
			]["values"].items()
			if value == unit_value
		][0]
		
		self.raw_unit_previous = raw_unit_name
	
	def value_as_unit(self, value) -> typ.Any:
		"""Return the given value expresse as the current internal unit,
		without the unit.
		"""
		
		if hasattr(self, "raw_value") and hasattr(self, "unit"):
			# (Guard) Value Compatibility
			if not self.is_compatible(value):
				msg = f"Tried setting socket ({self}) to incompatible value ({value}) of type {type(value)}"
				raise ValueError(msg)
			
			# Return Converted Unit
			return spu.convert_to(
				value, self.unit
			) / self.unit
		else:
			raise ValueError("Tried to get 'raw_value_as_unit', but class has no 'raw_value'")
	
	def _update_unit(self) -> None:
		"""Convert (if needed) the `raw_value` property, to use the unit
		set in the `unit` property.
		
		If the `raw_value` property isn't set, this only sets "unit_previous".
		
		Run right after setting the `unit` property, in order to synchronize
		the value with the new unit.
		"""
		if hasattr(self, "raw_value") and hasattr(self, "unit"):
			if hasattr(self.raw_value, "__getitem__"):
				self.raw_value = tuple(spu.convert_to(
					sp.Matrix(tuple(self.raw_value)) * self._unit_previous,
					self.unit,
				) / self.unit)
			else:
				self.raw_value = spu.convert_to(
					self.raw_value * self._unit_previous,
					self.unit,
				) / self.unit
		
		self._unit_previous = self.unit
	
	####################
	# - Callback Dispatcher
	####################
	def trigger_updates(self) -> None:
		if not self.is_output:
			self.node.update()
	
	####################
	# - Methods
	####################
	def is_compatible(self, value: typ.Any) -> bool:
		if not hasattr(self, "compatible_types"):
			return True
		
		for compatible_type, checks in self.compatible_types.items():
			if (
				compatible_type is typ.Any or
				isinstance(value, compatible_type)
			):
				return all(check(self, value) for check in checks)
		
		return False
	
	####################
	# - UI
	####################
	@classmethod
	def draw_color_simple(cls) -> contracts.BlenderColorRGB:
		return cls.socket_color
	
	def draw(
		self,
		context: bpy.types.Context,
		layout: bpy.types.UILayout,
		node: bpy.types.Node,
		text: str,
	) -> None:
		if self.is_output:
			self.draw_output(context, layout, node, text)
		else:
			self.draw_input(context, layout, node, text)
	
	def draw_input(
		self,
		context: bpy.types.Context,
		layout: bpy.types.UILayout,
		node: bpy.types.Node,
		text: str,
	) -> None:
		if self.is_linked:
			layout.label(text=text)
			return
		
		# Column
		col = layout.column(align=True)
		
		# Row: Label & Preview Toggle
		label_col_row = col.row(align=True)
		if hasattr(self, "draw_label_row"):
			self.draw_label_row(label_col_row, text)
		elif hasattr(self, "raw_unit"):
			label_col_row.label(text=text)
			label_col_row.prop(self, "raw_unit", text="")
		else:
			label_col_row.label(text=text)
		
		if hasattr(self, "draw_preview"):
			label_col_row.prop(
				self,
				"preview_active",
				toggle=True,
				text="",
				icon="SEQ_PREVIEW",
			)
		
		# Row: Preview (in Box)
		if hasattr(self, "draw_preview"):
			if self.preview_active:
				col_box = col.box()
				self.draw_preview(col_box)
		
		# Row(s): Value
		if hasattr(self, "draw_value"):
			self.draw_value(col)
		elif hasattr(self, "raw_value"):
			#col_row = col.row(align=True)
			col.prop(self, "raw_value", text="")
	
	def draw_output(
		self,
		context: bpy.types.Context,
		layout: bpy.types.UILayout,
		node: bpy.types.Node,
		text: str,
	) -> None:
		col = layout.column()
		row_col = col.row()
		row_col.alignment = "RIGHT"
		# Row: Label & Preview Toggle
		if hasattr(self, "draw_preview"):
			row_col.prop(
				self,
				"preview_active",
				toggle=True,
				text="",
				icon="SEQ_PREVIEW",
			)
		
		row_col.label(text=text)
		
		# Row: Preview (in box)
		if hasattr(self, "draw_preview"):
			if self.preview_active:
				col_box = col.box()
				self.draw_preview(col_box)
