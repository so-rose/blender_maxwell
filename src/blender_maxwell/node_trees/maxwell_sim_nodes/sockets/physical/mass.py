import bpy
import sympy.physics.units as spu

from .....utils.pydantic_sympy import SympyExpr
from ... import contracts as ct
from .. import base


####################
# - Blender Socket
####################
class PhysicalMassBLSocket(base.MaxwellSimSocket):
	socket_type = ct.SocketType.PhysicalMass
	bl_label = 'Mass'
	use_units = True

	####################
	# - Properties
	####################
	raw_value: bpy.props.FloatProperty(
		name='Unitless Mass',
		description='Represents the unitless part of mass',
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
class PhysicalMassSocketDef(base.SocketDef):
	socket_type: ct.SocketType = ct.SocketType.PhysicalMass

	default_unit: SympyExpr | None = None

	def init(self, bl_socket: PhysicalMassBLSocket) -> None:
		pass


####################
# - Blender Registration
####################
BL_REGISTER = [
	PhysicalMassBLSocket,
]
