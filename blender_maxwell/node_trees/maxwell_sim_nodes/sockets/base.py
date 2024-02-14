import typing as typ
import bpy

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
		
		# Declare Node Property: 'preset' EnumProperty
		if hasattr(cls, "draw_preview"):
			cls.__annotations__["preview_active"] = bpy.props.BoolProperty(
				name="Preview",
				description="Preview the socket value",
				default=False,
			)
	
	####################
	# - Methods
	####################
	def is_compatible(self, value: typ.Any) -> bool:
		for compatible_type, checks in self.compatible_types.items():
			if (
				compatible_type is typ.Any or
				isinstance(value, compatible_type)
			):
				return all(check(value) for check in checks)
		
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
	
	def draw_output(
		self,
		context: bpy.types.Context,
		layout: bpy.types.UILayout,
		node: bpy.types.Node,
		text: str,
	) -> None:
		layout.label(text=text)
