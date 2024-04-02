import bpy
import pydantic as pyd
import sympy.physics.units as spu

from .....utils.pydantic_sympy import SympyExpr
from ... import contracts as ct
from .. import base


####################
# - Blender Socket
####################
class PhysicalAngleBLSocket(base.MaxwellSimSocket):
	socket_type = ct.SocketType.PhysicalAngle
	bl_label = 'Physical Angle'
	use_units = True

	####################
	# - Properties
	####################
	raw_value: bpy.props.FloatProperty(
		name='Unitless Acceleration',
		description='Represents the unitless part of the acceleration',
		default=0.0,
		precision=4,
		update=(lambda self, context: self.sync_prop('raw_value', context)),
	)

	####################
	# - Socket UI
	####################
	def draw_value(self, col: bpy.types.UILayout) -> None:
		col.prop(self, 'raw_value', text='')

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
class PhysicalAngleSocketDef(pyd.BaseModel):
	socket_type: ct.SocketType = ct.SocketType.PhysicalAngle

	default_unit: SympyExpr | None = None

	def init(self, bl_socket: PhysicalAngleBLSocket) -> None:
		if self.default_unit:
			bl_socket.unit = self.default_unit


####################
# - Blender Registration
####################
BL_REGISTER = [
	PhysicalAngleBLSocket,
]
