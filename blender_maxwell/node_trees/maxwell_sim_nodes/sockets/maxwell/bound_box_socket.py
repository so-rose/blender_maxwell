import typing as typ

import bpy
import pydantic as pyd
import tidy3d as td

from .. import base
from ... import contracts

BOUND_FACE_ITEMS = [
	("PML", "PML", "Perfectly matched layer"),
	("PEC", "PEC", "Perfect electrical conductor"),
	("PMC", "PMC", "Perfect magnetic conductor"),
	("PERIODIC", "Periodic", "Infinitely periodic layer"),
]

class MaxwellBoundBoxBLSocket(base.BLSocket):
	socket_type = contracts.SocketType.MaxwellBoundBox
	bl_label = "Maxwell Bound Box"
	
	compatible_types = {
		td.BoundarySpec: {}
	}
	
	####################
	# - Properties
	####################
	x_pos: bpy.props.EnumProperty(
		name="+x Bound Face",
		description="+x choice of default boundary face",
		items=BOUND_FACE_ITEMS,
		default="PML",
		update=(lambda self, context: self.trigger_updates()),
	)
	x_neg: bpy.props.EnumProperty(
		name="-x Bound Face",
		description="-x choice of default boundary face",
		items=BOUND_FACE_ITEMS,
		default="PML",
		update=(lambda self, context: self.trigger_updates()),
	)
	y_pos: bpy.props.EnumProperty(
		name="+y Bound Face",
		description="+y choice of default boundary face",
		items=BOUND_FACE_ITEMS,
		default="PML",
		update=(lambda self, context: self.trigger_updates()),
	)
	y_neg: bpy.props.EnumProperty(
		name="-y Bound Face",
		description="-y choice of default boundary face",
		items=BOUND_FACE_ITEMS,
		default="PML",
		update=(lambda self, context: self.trigger_updates()),
	)
	z_pos: bpy.props.EnumProperty(
		name="+z Bound Face",
		description="+z choice of default boundary face",
		items=BOUND_FACE_ITEMS,
		default="PML",
		update=(lambda self, context: self.trigger_updates()),
	)
	z_neg: bpy.props.EnumProperty(
		name="-z Bound Face",
		description="-z choice of default boundary face",
		items=BOUND_FACE_ITEMS,
		default="PML",
		update=(lambda self, context: self.trigger_updates()),
	)
	
	####################
	# - UI
	####################
	def draw_value(self, col: bpy.types.UILayout) -> None:
		col.label(text="-/+ x")
		col_row = col.row(align=True)
		col_row.prop(self, "x_neg", text="")
		col_row.prop(self, "x_pos", text="")
		
		col.label(text="-/+ y")
		col_row = col.row(align=True)
		col_row.prop(self, "y_neg", text="")
		col_row.prop(self, "y_pos", text="")
		
		col.label(text="-/+ z")
		col_row = col.row(align=True)
		col_row.prop(self, "z_neg", text="")
		col_row.prop(self, "z_pos", text="")
	
	
	####################
	# - Computation of Default Value
	####################
	@property
	def default_value(self) -> td.BoundarySpec:
		return td.BoundarySpec()
	
	@default_value.setter
	def default_value(self, value: typ.Any) -> None:
		return None

####################
# - Socket Configuration
####################
class MaxwellBoundBoxSocketDef(pyd.BaseModel):
	socket_type: contracts.SocketType = contracts.SocketType.MaxwellBoundBox
	label: str
	
	def init(self, bl_socket: MaxwellBoundBoxBLSocket) -> None:
		pass

####################
# - Blender Registration
####################
BL_REGISTER = [
	MaxwellBoundBoxBLSocket,
]
