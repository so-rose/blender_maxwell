import typing as typ

import bpy
import sympy as sp
import pydantic as pyd

from .. import base
from ... import contracts

####################
# - Blender Socket
####################
class Real2DVectorBLSocket(base.BLSocket):
	socket_type = contracts.SocketType.Real2DVector
	bl_label = "Real2DVector"
	
	####################
	# - Properties
	####################
	raw_value: bpy.props.FloatVectorProperty(
		name="Unitless 2D Vector (global coordinate system)",
		description="Represents a real 2D (coordinate) vector",
		size=2,
		default=(0.0, 0.0),
		precision=4,
		update=(lambda self, context: self.trigger_updates()),
	)
	
	####################
	# - Computation of Default Value
	####################
	@property
	def default_value(self) -> sp.Expr:
		return tuple(self.raw_value)
	
	@default_value.setter
	def default_value(self, value: typ.Any) -> None:
		self.raw_value = tuple(value)

####################
# - Socket Configuration
####################
class Real2DVectorSocketDef(pyd.BaseModel):
	socket_type: contracts.SocketType = contracts.SocketType.Real2DVector
	label: str
	
	def init(self, bl_socket: Real2DVectorBLSocket) -> None:
		pass

####################
# - Blender Registration
####################
BL_REGISTER = [
	Real2DVectorBLSocket,
]
