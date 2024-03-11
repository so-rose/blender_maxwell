import typing as typ

import bpy
import sympy.physics.units as spu
import pydantic as pyd

from .....utils.pydantic_sympy import SympyExpr
from .. import base
from ... import contracts as ct

####################
# - Blender Socket
####################
class PhysicalTimeBLSocket(base.MaxwellSimSocket):
	socket_type = ct.SocketType.PhysicalTime
	bl_label = "Time"
	use_units = True
	
	####################
	# - Properties
	####################
	raw_value: bpy.props.FloatProperty(
		name="Unitless Time",
		description="Represents the unitless part of time",
		default=0.0,
		precision=4,
		update=(lambda self, context: self.sync_prop("raw_value", context)),
	)
	
	####################
	# - Socket UI
	####################
	def draw_value(self, col: bpy.types.UILayout) -> None:
		col.prop(self, "raw_value", text="")
	
	####################
	# - Default Value
	####################
	@property
	def value(self) -> SympyExpr:
		return self.raw_value * self.unit
	
	@value.setter
	def value(self, value: SympyExpr) -> None:
		self.raw_value = spu.convert_to(value, self.unit) / self.unit

####################
# - Socket Configuration
####################
class PhysicalTimeSocketDef(pyd.BaseModel):
	socket_type: ct.SocketType = ct.SocketType.PhysicalTime
	
	default_value: SympyExpr | None = None
	default_unit: typ.Any | None = None
	
	def init(self, bl_socket: PhysicalTimeBLSocket) -> None:
		if self.default_value:
			bl_socket.value = self.default_value
		if self.default_unit:
			bl_socket.unit = self.default_unit

####################
# - Blender Registration
####################
BL_REGISTER = [
	PhysicalTimeBLSocket,
]
