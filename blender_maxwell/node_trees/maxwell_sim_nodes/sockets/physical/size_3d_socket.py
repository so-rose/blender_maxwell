import typing as typ

import bpy
import sympy as sp
import sympy.physics.units as spu
import pydantic as pyd

from .. import base
from ... import contracts

class PhysicalSize3DBLSocket(base.BLSocket):
	socket_type = contracts.SocketType.PhysicalSize3D
	bl_label = "Physical Volume"
	use_units = True
	
	compatible_types = {
		sp.Expr: {
			lambda self, v: v.is_real,
			lambda self, v: len(v.free_symbols) == 0,
			lambda self, v: any(
				contracts.is_exactly_expressed_as_unit(v, unit)
				for unit in self.units.values()
			)
		},
	}
	
	####################
	# - Properties
	####################
	raw_value: bpy.props.FloatVectorProperty(
		name="Unitless 3D Size",
		description="Represents the unitless part of the 3D size",
		size=3,
		default=(1.0, 1.0, 1.0),
		precision=4,
		update=(lambda self, context: self.trigger_updates()),
	)
	
	####################
	# - Computation of Default Value
	####################
	@property
	def default_value(self) -> sp.Expr:
		return sp.Matrix(tuple(self.raw_value)) * self.unit
	
	@default_value.setter
	def default_value(self, value: typ.Any) -> None:
		self.raw_value = self.value_as_unit(value)

####################
# - Socket Configuration
####################
class PhysicalSize3DSocketDef(pyd.BaseModel):
	socket_type: contracts.SocketType = contracts.SocketType.PhysicalSize3D
	label: str
	
	default_unit: typ.Any | None = None
	
	def init(self, bl_socket: PhysicalSize3DBLSocket) -> None:
		if self.default_unit:
			bl_socket.unit = self.default_unit

####################
# - Blender Registration
####################
BL_REGISTER = [
	PhysicalSize3DBLSocket,
]
