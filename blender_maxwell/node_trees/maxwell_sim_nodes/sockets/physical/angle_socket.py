import typing as typ

import bpy
import pydantic as pyd

from .. import base
from ... import contracts

####################
# - Blender Socket
####################
class PhysicalAngleBLSocket(base.BLSocket):
	socket_type = contracts.SocketType.PhysicalAngle
	bl_label = "PhysicalAngle"
	use_units = True
	
	####################
	# - Properties
	####################
	raw_value: bpy.props.FloatProperty(
		name="Unitless Acceleration",
		description="Represents the unitless part of the acceleration",
		default=0.0,
		precision=4,
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
class PhysicalAngleSocketDef(pyd.BaseModel):
	socket_type: contracts.SocketType = contracts.SocketType.PhysicalAngle
	label: str
	
	default_unit: typ.Any | None = None
	
	def init(self, bl_socket: PhysicalAngleBLSocket) -> None:
		if self.default_unit:
			bl_socket.unit = self.default_unit

####################
# - Blender Registration
####################
BL_REGISTER = [
	PhysicalAngleBLSocket,
]
