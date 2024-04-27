import typing as typ

import bpy
import pydantic as pyd
import sympy as sp

from blender_maxwell.utils import bl_cache, logger
from blender_maxwell.utils import extra_sympy_units as spux

from ... import contracts as ct
from .. import base

log = logger.get(__name__)


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

	int_symbols: frozenset[spux.IntSymbol] = bl_cache.BLField(frozenset())
	real_symbols: frozenset[spux.RealSymbol] = bl_cache.BLField(frozenset())
	complex_symbols: frozenset[spux.ComplexSymbol] = bl_cache.BLField(frozenset())

	@bl_cache.cached_bl_property(persist=False)
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
		if len(self.symbols) > 0:
			box = col.box()
			split = box.split(factor=0.3)

			# Left Col
			col = split.column()
			col.label(text='Let:')

			# Right Col
			col = split.column()
			col.alignment = 'RIGHT'
			for sym in self.symbols:
				col.label(text=spux.pretty_symbol(sym))

	####################
	# - Computation of Default Value
	####################
	@property
	def value(self) -> sp.Expr:
		return sp.sympify(
			self.raw_value,
			locals={sym.name: sym for sym in self.symbols},
			strict=False,
			convert_xor=True,
		).subs(spux.ALL_UNIT_SYMBOLS)

	@value.setter
	def value(self, value: str) -> None:
		self.raw_value = sp.sstr(value)

	@property
	def lazy_value_func(self) -> ct.LazyValueFuncFlow:
		return ct.LazyValueFuncFlow(
			func=sp.lambdify(self.symbols, self.value, 'jax'),
			func_args=[spux.sympy_to_python_type(sym) for sym in self.symbols],
			supports_jax=True,
		)


####################
# - Socket Configuration
####################
class ExprSocketDef(base.SocketDef):
	socket_type: ct.SocketType = ct.SocketType.Expr

	int_symbols: frozenset[spux.IntSymbol] = frozenset()
	real_symbols: frozenset[spux.RealSymbol] = frozenset()
	complex_symbols: frozenset[spux.ComplexSymbol] = frozenset()

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

	# Expression
	default_expr: spux.SympyExpr = sp.S(1)
	allow_units: bool = True

	@pyd.model_validator(mode='after')
	def check_default_expr_follows_unit_allowance(self) -> typ.Self:
		"""Checks that `self.default_expr` only uses units if `self.allow_units` is defined.

		Raises:
			ValueError: If the expression uses symbols not defined in `self.symbols`.
		"""
		if spux.uses_units(self.default_expr) and not self.allow_units:
			msg = f'Expression {self.default_expr} uses units, but "self.allow_units" is False'
			raise ValueError(msg)

		return self

	## TODO: Validator for Symbol Usage

	def init(self, bl_socket: ExprBLSocket) -> None:
		bl_socket.value = self.default_expr
		bl_socket.int_symbols = self.int_symbols
		bl_socket.real_symbols = self.real_symbols
		bl_socket.complex_symbols = self.complex_symbols


####################
# - Blender Registration
####################
BL_REGISTER = [
	ExprBLSocket,
]
