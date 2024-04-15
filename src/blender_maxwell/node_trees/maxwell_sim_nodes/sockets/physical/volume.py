import bpy
import sympy.physics.units as spu

from .....utils.pydantic_sympy import SympyExpr
from ... import contracts as ct
from .. import base


class PhysicalVolumeBLSocket(base.MaxwellSimSocket):
	socket_type = ct.SocketType.PhysicalVolume
	bl_label = 'Volume'
	use_units = True

	####################
	# - Properties
	####################
	raw_value: bpy.props.FloatProperty(
		name='Unitless Volume',
		description='Represents the unitless part of the area',
		default=0.0,
		precision=6,
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
class PhysicalVolumeSocketDef(base.SocketDef):
	socket_type: ct.SocketType = ct.SocketType.PhysicalVolume

	default_unit: SympyExpr | None = None

	def init(self, bl_socket: PhysicalVolumeBLSocket) -> None:
		if self.default_unit:
			bl_socket.unit = self.default_unit


####################
# - Blender Registration
####################
BL_REGISTER = [
	PhysicalVolumeBLSocket,
]
