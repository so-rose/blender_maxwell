import typing as typ

import bpy
import pydantic as pyd
import sympy as sp

from blender_maxwell.utils import bl_cache
from blender_maxwell.utils import extra_sympy_units as spux

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
		update=(lambda self, context: self.on_prop_changed('raw_value', context)),
	)

	int_symbols: set[spux.IntSymbol] = bl_cache.BLField([])
	real_symbols: set[spux.RealSymbol] = bl_cache.BLField([])
	complex_symbols: set[spux.ComplexSymbol] = bl_cache.BLField([])

	@property
	def symbols(self) -> list[spux.Symbol]:
		"""Retrieves all symbols by concatenating int, real, and complex symbols, and sorting them by name.

		The order is guaranteed to be **deterministic**.

		Returns:
			All symbols valid for use in the expression.
		"""
		return sorted(
			self.int_symbols | self.real_symbols | self.complex_symbols,
			key=lambda sym: sym.name,
		)

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
		expr = sp.sympify(
			self.raw_value,
			locals={sym.name: sym for sym in self.symbols},
			strict=False,
			convert_xor=True,
		).subs(spux.ALL_UNIT_SYMBOLS)

		if not expr.free_symbols.issubset(self.symbols):
			msg = f'Expression "{expr}" (symbols={self.expr.free_symbols}) has invalid symbols (valid symbols: {self.symbols})'
			raise ValueError(msg)

		return expr

	@value.setter
	def value(self, value: str) -> None:
		self.raw_value = sp.sstr(value)

	@property
	def lazy_value_func(self) -> ct.LazyValueFuncFlow:
		return ct.LazyValueFuncFlow(
			func=sp.lambdify(self.symbols, self.value, 'jax'),
			func_args=[
				(sym.name, spux.sympy_to_python_type(sym)) for sym in self.symbols
			],
			supports_jax=True,
		)


####################
# - Socket Configuration
####################
class ExprSocketDef(base.SocketDef):
	socket_type: ct.SocketType = ct.SocketType.Expr

	_x = sp.Symbol('x', real=True)
	int_symbols: list[spux.IntSymbol] = []
	real_symbols: list[spux.RealSymbol] = [_x]
	complex_symbols: list[spux.ComplexSymbol] = []

	# Expression
	default_expr: spux.SympyExpr = _x
	allow_units: bool = True

	@pyd.model_validator(mode='after')
	def check_default_expr_follows_unit_allowance(self) -> typ.Self:
		"""Checks that `self.default_expr` only uses units if `self.allow_units` is defined.

		Raises:
			ValueError: If the expression uses symbols not defined in `self.symbols`.
		"""
		if not spux.uses_units(self.default_expr):
			msg = f'Expression symbols ({self.default_expr.free_symbol}) are not a strict subset of defined symbols ({self.symbols})'
			raise ValueError(msg)

	@pyd.model_validator(mode='after')
	def check_default_expr_uses_allowed_symbols(self) -> typ.Self:
		"""Checks that `self.default_expr` only uses symbols defined in `self.symbols`.

		Raises:
			ValueError: If the expression uses symbols not defined in `self.symbols`.
		"""
		if not self.default_expr.free_symbols.issubset(self.symbols):
			msg = f'Expression symbols ({self.default_expr.free_symbol}) are not a strict subset of defined symbols ({self.symbols})'
			raise ValueError(msg)

	def init(self, bl_socket: ExprBLSocket) -> None:
		bl_socket.value = self.default_expr
		bl_socket.symbols = self.symbols


####################
# - Blender Registration
####################
BL_REGISTER = [
	ExprBLSocket,
]
