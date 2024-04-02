
import bpy
import pydantic as pyd
import sympy as sp

from .....utils.pydantic_sympy import SympyExpr
from ... import contracts as ct
from .. import base


####################
# - Blender Socket
####################
class RationalNumberBLSocket(base.MaxwellSimSocket):
	socket_type = ct.SocketType.RationalNumber
	bl_label = 'Rational Number'

	####################
	# - Properties
	####################
	raw_value: bpy.props.IntVectorProperty(
		name='Rational Number',
		description='Represents a rational number (int / int)',
		size=2,
		default=(1, 1),
		subtype='NONE',
		update=(lambda self, context: self.sync_prop('raw_value', context)),
	)

	####################
	# - UI
	####################
	def draw_value(self, col: bpy.types.UILayout) -> None:
		col_row = col.row(align=True)
		col_row.prop(self, 'raw_value', text='')

	####################
	# - Default Value
	####################
	@property
	def value(self) -> sp.Rational:
		p, q = self.raw_value
		return sp.Rational(p, q)

	@value.setter
	def value(self, value: float | tuple[int, int] | SympyExpr) -> None:
		if isinstance(value, float):
			approx_rational = sp.nsimplify(value)
			self.raw_value = (approx_rational.p, approx_rational.q)
		elif isinstance(value, tuple):
			self.raw_value = value
		else:
			self.raw_value = (value.p, value.q)


####################
# - Socket Configuration
####################
class RationalNumberSocketDef(pyd.BaseModel):
	socket_type: ct.SocketType = ct.SocketType.RationalNumber

	default_value: SympyExpr = sp.Rational(0, 1)

	def init(self, bl_socket: RationalNumberBLSocket) -> None:
		bl_socket.value = self.default_value


####################
# - Blender Registration
####################
BL_REGISTER = [
	RationalNumberBLSocket,
]
