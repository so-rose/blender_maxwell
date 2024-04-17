import bpy
import sympy as sp

from blender_maxwell.utils import extra_sympy_units as spux
from blender_maxwell.utils.pydantic_sympy import SympyExpr

from ... import bl_cache
from ... import contracts as ct
from .. import base


class ExprBLSocket(base.MaxwellSimSocket):
	socket_type = ct.SocketType.Expr
	bl_label = 'Expr'

	####################
	# - Properties
	####################
	raw_value: bpy.props.StringProperty(
		name='Expr',
		description='Represents a symbolic expression',
		default='',
		update=(lambda self, context: self.sync_prop('raw_value', context)),
	)

	symbols: list[sp.Symbol] = bl_cache.BLField([])
	## TODO: Way of assigning assumptions to symbols.
	## TODO: Dynamic add/remove of symbols

	####################
	# - Socket UI
	####################
	def draw_value(self, col: bpy.types.UILayout) -> None:
		col.prop(self, 'raw_value', text='')

	####################
	# - Computation of Default Value
	####################
	@property
	def value(self) -> sp.Expr:
		return sp.sympify(
			self.raw_value,
			strict=False,
			convert_xor=True,
		).subs(spux.ALL_UNIT_SYMBOLS)

	@value.setter
	def value(self, value: str) -> None:
		self.raw_value = str(value)

	@property
	def lazy_value(self) -> sp.Expr:
		return ct.LazyDataValue.from_function(
			sp.lambdify(self.symbols, self.value, 'jax'),
			free_args=(tuple(str(sym) for sym in self.symbols), frozenset()),
			supports_jax=True,
		)


####################
# - Socket Configuration
####################
class ExprSocketDef(base.SocketDef):
	socket_type: ct.SocketType = ct.SocketType.Expr

	_x = sp.Symbol('x', real=True)
	symbols: list[SympyExpr] = [_x]
	default_expr: SympyExpr = _x

	def init(self, bl_socket: ExprBLSocket) -> None:
		bl_socket.value = self.default_expr
		bl_socket.symbols = self.symbols


####################
# - Blender Registration
####################
BL_REGISTER = [
	ExprBLSocket,
]
