import typing as typ

import bpy
import pydantic as pyd

from .. import base
from ... import contracts

####################
# - Blender Socket
####################
class PhysicalAccelScalarBLSocket(base.BLSocket):
	socket_type = contracts.SocketType.PhysicalAccelScalar
	bl_label = "PhysicalAccel"
	use_units = True
	
	####################
	# - Properties
	####################
	raw_value: bpy.props.FloatProperty(
		name="Unitless Acceleration",
		description="Represents the unitless part of the acceleration",
		default=0.0,
		precision=6,
		update=(lambda self, context: self.trigger_updates()),
	)
	
	####################
	# - Default Value
	####################
	@property
	def default_value(self) -> None:
		return self.raw_value * self.unit
	
	@default_value.setter
	def default_value(self, value: typ.Any) -> None:
		self.raw_value = self.value_as_unit(value)

####################
# - Socket Configuration
####################
class PhysicalAccelScalarSocketDef(pyd.BaseModel):
	socket_type: contracts.SocketType = contracts.SocketType.PhysicalAccelScalar
	label: str
	
	default_unit: typ.Any | None = None
	
	def init(self, bl_socket: PhysicalAccelScalarBLSocket) -> None:
		if self.default_unit:
			bl_socket.unit = self.default_unit

####################
# - Blender Registration
####################
BL_REGISTER = [
	PhysicalAccelScalarBLSocket,
]
