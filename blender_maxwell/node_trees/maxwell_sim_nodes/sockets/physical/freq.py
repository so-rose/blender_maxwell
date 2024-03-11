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
class PhysicalFreqBLSocket(base.MaxwellSimSocket):
	socket_type = ct.SocketType.PhysicalFreq
	bl_label = "Frequency"
	use_units = True
	
	####################
	# - Properties
	####################
	raw_value: bpy.props.FloatProperty(
		name="Unitless Frequency",
		description="Represents the unitless part of the frequency",
		default=0.0,
		precision=6,
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
class PhysicalFreqSocketDef(pyd.BaseModel):
	socket_type: ct.SocketType = ct.SocketType.PhysicalFreq
	
	default_value: SympyExpr | None = None
	default_unit: SympyExpr | None = None
	
	def init(self, bl_socket: PhysicalFreqBLSocket) -> None:
		if self.default_value:
			bl_socket.value = self.default_value
		if self.default_unit:
			bl_socket.unit = self.default_unit

####################
# - Blender Registration
####################
BL_REGISTER = [
	PhysicalFreqBLSocket,
]
