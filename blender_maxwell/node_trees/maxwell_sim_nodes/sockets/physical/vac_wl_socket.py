import typing as typ

import bpy
import pydantic as pyd

from .. import base
from ... import contracts

####################
# - Blender Socket
####################
class PhysicalVacWLBLSocket(base.BLSocket):
	socket_type = contracts.SocketType.PhysicalVacWL
	bl_label = "PhysicalVacWL"
	use_units = True
	
	####################
	# - Properties
	####################
	raw_value: bpy.props.FloatProperty(
		name="Unitless Vacuum Wavelength",
		description="Represents the unitless part of the vacuum wavelength",
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
class PhysicalVacWLSocketDef(pyd.BaseModel):
	socket_type: contracts.SocketType = contracts.SocketType.PhysicalVacWL
	label: str
	
	def init(self, bl_socket: PhysicalVacWLBLSocket) -> None:
		pass

####################
# - Blender Registration
####################
BL_REGISTER = [
	PhysicalVacWLBLSocket,
]
