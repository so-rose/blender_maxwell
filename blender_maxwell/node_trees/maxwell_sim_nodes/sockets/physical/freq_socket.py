import typing as typ

import bpy
import pydantic as pyd

from .. import base
from ... import contracts

####################
# - Blender Socket
####################
class PhysicalFreqBLSocket(base.BLSocket):
	socket_type = contracts.SocketType.PhysicalFreq
	bl_label = "PhysicalFreq"
	use_units = True
	
	####################
	# - Properties
	####################
	raw_value: bpy.props.FloatProperty(
		name="Unitless Frequency",
		description="Represents the unitless part of the frequency",
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
class PhysicalFreqSocketDef(pyd.BaseModel):
	socket_type: contracts.SocketType = contracts.SocketType.PhysicalFreq
	label: str
	
	def init(self, bl_socket: PhysicalFreqBLSocket) -> None:
		pass

####################
# - Blender Registration
####################
BL_REGISTER = [
	PhysicalFreqBLSocket,
]
