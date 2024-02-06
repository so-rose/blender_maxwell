import bpy

from . import constants, types

class MaxwellSourceSocket(bpy.types.NodeSocket):
	bl_idname = types.MaxwellSourceSocketType
	bl_label = "Maxwell Source"

	def draw(self, context, layout, node, text):
		layout.label(text=text)
	
	@classmethod
	def draw_color_simple(cls):
		return constants.COLOR_SOCKET_SOURCE


class MaxwellMediumSocket(bpy.types.NodeSocket):
	bl_idname = types.MaxwellMediumSocketType
	bl_label = "Maxwell Medium"

	def draw(self, context, layout, node, text):
		layout.label(text=text)
	
	@classmethod
	def draw_color_simple(cls):
		return constants.COLOR_SOCKET_MEDIUM

class MaxwellStructureSocket(bpy.types.NodeSocket):
	bl_idname = types.MaxwellStructureSocketType
	bl_label = "Maxwell Structure"
	
	def draw(self, context, layout, node, text):
		layout.label(text=text)
	
	@classmethod
	def draw_color_simple(cls):
		return constants.COLOR_SOCKET_STRUCTURE

class MaxwellBoundSocket(bpy.types.NodeSocket):
	bl_idname = types.MaxwellBoundSocketType
	bl_label = "Maxwell Bound"
	
	def draw(self, context, layout, node, text):
		layout.label(text=text)
	
	@classmethod
	def draw_color_simple(cls):
		return constants.COLOR_SOCKET_BOUND

class MaxwellFDTDSimSocket(bpy.types.NodeSocket):
	bl_idname = types.MaxwellFDTDSimSocketType
	bl_label = "Maxwell Bound"
	
	def draw(self, context, layout, node, text):
		layout.label(text=text)
	
	@classmethod
	def draw_color_simple(cls):
		return constants.COLOR_SOCKET_FDTDSIM



####################
# - Blender Registration
####################
BL_REGISTER = [
	MaxwellSourceSocket,
	MaxwellMediumSocket,
	MaxwellStructureSocket,
	MaxwellBoundSocket,
]
