import typing as typ

import bpy
import sympy as sp
import sympy.physics.units as spu
import pydantic as pyd

from .. import base
from ... import contracts

class PhysicalPoint3DBLSocket(base.BLSocket):
	socket_type = contracts.SocketType.PhysicalPoint3D
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
		name="Unitless 3D Point (global coordinate system)",
		description="Represents the unitless part of the 3D point",
		size=3,
		default=(0.0, 0.0, 0.0),
		precision=4,
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
class PhysicalPoint3DSocketDef(pyd.BaseModel):
	socket_type: contracts.SocketType = contracts.SocketType.PhysicalPoint3D
	label: str
	
	default_unit: typ.Any | None = None
	
	def init(self, bl_socket: PhysicalPoint3DBLSocket) -> None:
		if self.default_unit:
			bl_socket.unit = self.default_unit

####################
# - Blender Registration
####################
BL_REGISTER = [
	PhysicalPoint3DBLSocket,
]
