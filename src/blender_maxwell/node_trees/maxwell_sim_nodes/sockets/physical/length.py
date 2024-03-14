import typing as typ

import bpy
import sympy.physics.units as spu
import numpy as np
import pydantic as pyd

from .....utils.pydantic_sympy import SympyExpr
from .. import base
from ... import contracts as ct

####################
# - Blender Socket
####################
class PhysicalLengthBLSocket(base.MaxwellSimSocket):
	socket_type = ct.SocketType.PhysicalLength
	bl_label = "PhysicalLength"
	use_units = True
	
	####################
	# - Properties
	####################
	raw_value: bpy.props.FloatProperty(
		name="Unitless Force",
		description="Represents the unitless part of the force",
		default=0.0,
		precision=6,
		update=(lambda self, context: self.sync_prop("raw_value", context)),
	)
	
	min_len: bpy.props.FloatProperty(
		name="Min Length",
		description="Lowest length",
		default=0.0,
		precision=4,
		update=(lambda self, context: self.sync_prop("min_len", context)),
	)
	max_len: bpy.props.FloatProperty(
		name="Max Length",
		description="Highest length",
		default=0.0,
		precision=4,
		update=(lambda self, context: self.sync_prop("max_len", context)),
	)
	steps: bpy.props.IntProperty(
		name="Length Steps",
		description="# of steps between min and max",
		default=2,
		update=(lambda self, context: self.sync_prop("steps", context)),
	)
	
	####################
	# - Socket UI
	####################
	def draw_value(self, col: bpy.types.UILayout) -> None:
		col.prop(self, "raw_value", text="")
	
	def draw_value_list(self, col: bpy.types.UILayout) -> None:
		col.prop(self, "min_len", text="Min")
		col.prop(self, "max_len", text="Max")
		col.prop(self, "steps", text="Steps")
	
	####################
	# - Default Value
	####################
	@property
	def value(self) -> SympyExpr:
		return self.raw_value * self.unit
	@value.setter
	def value(self, value: SympyExpr) -> None:
		self.raw_value = spu.convert_to(value, self.unit) / self.unit
	
	@property
	def value_list(self) -> list[SympyExpr]:
		return [
			el * self.unit
			for el in np.linspace(self.min_len, self.max_len, self.steps)
		]
	@value_list.setter
	def value_list(self, value: tuple[SympyExpr, SympyExpr, int]):
		self.min_len, self.max_len, self.steps = [
			spu.convert_to(el, self.unit) / self.unit
			for el in value[:2]
		] + [value[2]]
	
	def sync_unit_change(self) -> None:
		if self.is_list:
			self.value_list = (
				spu.convert_to(
					self.min_len * self.prev_unit,
					self.unit
				),
				spu.convert_to(
					self.max_len * self.prev_unit,
					self.unit
				),
				self.steps,
			)
		else:
			self.value = self.value / self.unit * self.prev_unit
		
		self.prev_active_unit = self.active_unit

####################
# - Socket Configuration
####################
class PhysicalLengthSocketDef(pyd.BaseModel):
	socket_type: ct.SocketType = ct.SocketType.PhysicalLength
	
	default_value: SympyExpr = 1*spu.um
	default_unit: SympyExpr | None = None
	is_list: bool = False
	
	min_len: SympyExpr = 400.0*spu.nm
	max_len: SympyExpr = 600.0*spu.nm
	steps: SympyExpr = 50
	
	def init(self, bl_socket: PhysicalLengthBLSocket) -> None:
		bl_socket.value = self.default_value
		bl_socket.is_list = self.is_list
		
		if self.default_unit:
			bl_socket.unit = self.default_unit
		
		if self.is_list:
			bl_socket.value_list = (self.min_len, self.max_len, self.steps)

####################
# - Blender Registration
####################
BL_REGISTER = [
	PhysicalLengthBLSocket,
]
