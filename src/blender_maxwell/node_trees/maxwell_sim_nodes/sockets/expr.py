# blender_maxwell
# Copyright (C) 2024 blender_maxwell Project Contributors
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""Implements the `ExprSocket` node socket."""

import enum
import typing as typ

import bpy
import pydantic as pyd
import sympy as sp

from blender_maxwell.utils import bl_cache, logger, sim_symbols
from blender_maxwell.utils import extra_sympy_units as spux

from .. import contracts as ct
from . import base

log = logger.get(__name__)

Int2: typ.TypeAlias = tuple[int, int]
Int3: typ.TypeAlias = tuple[int, int, int]
Int22: typ.TypeAlias = tuple[tuple[int, int], tuple[int, int]]
Int32: typ.TypeAlias = tuple[tuple[int, int], tuple[int, int], tuple[int, int]]
Float2: typ.TypeAlias = tuple[float, float]
Float3: typ.TypeAlias = tuple[float, float, float]
Float22: typ.TypeAlias = tuple[tuple[float, float], tuple[float, float]]
Float32: typ.TypeAlias = tuple[
	tuple[float, float], tuple[float, float], tuple[float, float]
]


####################
# - Utilitives
####################
def unicode_superscript(n: int) -> str:
	"""Transform an integer into its unicode-based superscript character."""
	return ''.join(['⁰¹²³⁴⁵⁶⁷⁸⁹'[ord(c) - ord('0')] for c in str(n)])


def _check_sym_oo(sym):
	return sym.is_real or sym.is_rational or sym.is_integer


class InfoDisplayCol(enum.StrEnum):
	"""Valid columns for specifying displayed information from an `ct.InfoFlow`."""

	Length = enum.auto()
	MathType = enum.auto()
	Unit = enum.auto()

	@staticmethod
	def to_name(value: typ.Self) -> str:
		IDC = InfoDisplayCol
		return {
			IDC.Length: 'L',
			IDC.MathType: '∈',
			IDC.Unit: 'U',
		}[value]

	@staticmethod
	def to_icon(value: typ.Self) -> str:
		IDC = InfoDisplayCol
		return {
			IDC.Length: '',
			IDC.MathType: '',
			IDC.Unit: '',
		}[value]


####################
# - Socket
####################
class ExprBLSocket(base.MaxwellSimSocket):
	"""The `Expr` ("Expression") socket is an accessible approach to specifying any expression.

	Attributes:
		size: The dimensionality of the expression.
			The socket can exposes a UI for scalar, 2D, and 3D.
			Otherwise, a string-based `sympy` expression is the fallback.
		mathtype: The mathematical identity of the expression.
			Encompasses the usual suspects ex. integer, rational, real, complex, etc. .
			Generally, there is a UI available for all of these.
			The enum itself can be dynamically altered, ex. via its UI dropdown support.
		physical_type: The physical identity of the expression.
			The default indicator of a unitless (aka. non-physical) expression is `spux.PhysicalType.NonPhysical`.
			When active, `self.active_unit` can be used via the UI to select valid unit of the given `self.physical_type`, and `self.unit` works.
			The enum itself can be dynamically altered, ex. via its UI dropdown support.
		symbols: The symbolic variables valid in the context of the expression.
			Various features, including `Func` support, become available when symbols are in use.
			The presence of symbols forces fallback to a string-based `sympy` expression UI.

		active_unit: The currently active unit, as a dropdown.
			Its values are always the valid units of the currently active `physical_type`.
	"""

	socket_type = ct.SocketType.Expr
	bl_label = 'Expr'

	####################
	# - Properties
	####################
	size: spux.NumberSize1D = bl_cache.BLField(spux.NumberSize1D.Scalar)
	mathtype: spux.MathType = bl_cache.BLField(spux.MathType.Real)
	physical_type: spux.PhysicalType = bl_cache.BLField(spux.PhysicalType.NonPhysical)

	# Symbols
	output_name: sim_symbols.SimSymbolName = bl_cache.BLField(
		sim_symbols.SimSymbolName.Expr
	)
	active_symbols: list[sim_symbols.SimSymbol] = bl_cache.BLField([])

	@property
	def symbols(self) -> set[sp.Symbol]:
		"""Current symbols as an unordered set."""
		return {sim_symbol.sp_symbol for sim_symbol in self.active_symbols}

	@bl_cache.cached_bl_property(depends_on={'symbols'})
	def sorted_symbols(self) -> list[sp.Symbol]:
		"""Current symbols as a sorted list."""
		return sorted(self.symbols, key=lambda sym: sym.name)

	# Unit
	active_unit: enum.StrEnum = bl_cache.BLField(
		enum_cb=lambda self, _: self.search_valid_units(),
		cb_depends_on={'physical_type'},
	)

	def search_valid_units(self) -> list[ct.BLEnumElement]:
		"""Compute Blender enum elements of valid units for the current `physical_type`."""
		if self.physical_type is not spux.PhysicalType.NonPhysical:
			return [
				(sp.sstr(unit), spux.sp_to_str(unit), sp.sstr(unit), '', i)
				for i, unit in enumerate(self.physical_type.valid_units)
			]
		return []

	# UI: Value
	## Expression
	raw_value_spstr: str = bl_cache.BLField('0.0')
	## 1D
	raw_value_int: int = bl_cache.BLField(0)
	raw_value_rat: Int2 = bl_cache.BLField((0, 1))
	raw_value_float: float = bl_cache.BLField(0.0, float_prec=4)
	raw_value_complex: Float2 = bl_cache.BLField((0.0, 0.0))
	## 2D
	raw_value_int2: Int2 = bl_cache.BLField((0, 0))
	raw_value_rat2: Int22 = bl_cache.BLField(((0, 1), (0, 1)))
	raw_value_float2: Float2 = bl_cache.BLField((0.0, 0.0), float_prec=4)
	raw_value_complex2: Float22 = bl_cache.BLField(
		((0.0, 0.0), (0.0, 0.0)), float_prec=4
	)
	## 3D
	raw_value_int3: Int3 = bl_cache.BLField((0, 0, 0))
	raw_value_rat3: Int32 = bl_cache.BLField(((0, 1), (0, 1), (0, 1)))
	raw_value_float3: Float3 = bl_cache.BLField((0.0, 0.0, 0.0), float_prec=4)
	raw_value_complex3: Float32 = bl_cache.BLField(
		((0.0, 0.0), (0.0, 0.0), (0.0, 0.0)), float_prec=4
	)

	# UI: Range
	steps: int = bl_cache.BLField(2, soft_min=2, abs_min=0)
	scaling: ct.ScalingMode = bl_cache.BLField(ct.ScalingMode.Lin)
	## Expression
	raw_min_spstr: str = bl_cache.BLField('0.0')
	raw_max_spstr: str = bl_cache.BLField('1.0')
	## By MathType
	raw_range_int: Int2 = bl_cache.BLField((0, 1))
	raw_range_rat: Int22 = bl_cache.BLField(((0, 1), (1, 1)))
	raw_range_float: Float2 = bl_cache.BLField((0.0, 1.0))
	raw_range_complex: Float22 = bl_cache.BLField(
		((0.0, 0.0), (1.0, 1.0)), float_prec=4
	)

	# UI: Info
	show_func_ui: bool = bl_cache.BLField(True)
	show_info_columns: bool = bl_cache.BLField(False)
	info_columns: set[InfoDisplayCol] = bl_cache.BLField(
		{InfoDisplayCol.Length, InfoDisplayCol.MathType}
	)

	####################
	# - Computed String Expressions
	####################
	@bl_cache.cached_bl_property(depends_on={'raw_value_spstr'})
	def raw_value_sp(self) -> spux.SympyExpr:
		return self._parse_expr_str(self.raw_value_spstr)

	@bl_cache.cached_bl_property(depends_on={'raw_min_spstr'})
	def raw_min_sp(self) -> spux.SympyExpr:
		return self._parse_expr_str(self.raw_min_spstr)

	@bl_cache.cached_bl_property(depends_on={'raw_max_spstr'})
	def raw_max_sp(self) -> spux.SympyExpr:
		return self._parse_expr_str(self.raw_max_spstr)

	####################
	# - Computed Unit
	####################
	@bl_cache.cached_bl_property(depends_on={'active_unit'})
	def unit(self) -> spux.Unit | None:
		"""Gets the current active unit.

		Returns:
			The current active `sympy` unit.

			If the socket expression is unitless, this returns `None`.
		"""
		if self.active_unit is not None:
			return spux.unit_str_to_unit(self.active_unit)

		return None

	prev_unit: str | None = bl_cache.BLField(None)

	####################
	# - Prop-Change Callback
	####################
	def on_socket_prop_changed(self, prop_name: str) -> None:
		# Conditional Unit-Conversion
		## -> This is niche functionality, but the only way to convert units.
		## -> We can only catch 'unit' since it's at the end of a depschain.
		if prop_name == 'unit':
			# Check Unit Change
			## -> self.prev_unit only updates here; "lags" behind self.unit.
			## -> 1. "Laggy" unit must be different than new unit.
			## -> 2. Unit-conversion of value only within same physical_type
			## -> 3. Never unit-convert expressions w/symbolic variables
			## No matter what, prev_unit is always regenerated.
			prev_unit = (
				spux.unit_str_to_unit(self.prev_unit)
				if self.prev_unit is not None
				else None
			)
			if (
				prev_unit != self.unit
				and prev_unit in self.physical_type.valid_units
				and not self.symbols
			):
				self.value = self.value.subs({self.unit: prev_unit})
				self.lazy_range = self.lazy_range.correct_unit(prev_unit)

			self.prev_unit = self.active_unit

	####################
	# - Value Utilities
	####################
	def _parse_expr_info(
		self, expr: spux.SympyExpr
	) -> tuple[spux.MathType, tuple[int, ...] | None, spux.UnitDimension]:
		"""Parse a given expression for mathtype and size information.

		Various compatibility checks are also performed, allowing this method to serve as a generic runtime validator/parser for any expressions that need to enter the socket.
		"""
		# Parse MathType
		mathtype = spux.MathType.from_expr(expr)
		if not self.mathtype.is_compatible(mathtype):
			msg = f'MathType is {self.mathtype}, but tried to set expr {expr} with mathtype {mathtype}'
			raise ValueError(msg)

		# Parse Symbols
		if expr.free_symbols and not expr.free_symbols.issubset(self.symbols):
			msg = f'Tried to set expr {expr} with free symbols {expr.free_symbols}, which is incompatible with socket symbols {self.symbols}'
			raise ValueError(msg)

		# Parse Dimensions
		shape = spux.parse_shape(expr)
		if not self.size.supports_shape(shape):
			msg = f'Expr {expr} has non-1D shape {shape}, which is incompatible with the expr socket (shape {self.shape})'
			raise ValueError(msg)

		size = spux.NumberSize1D.from_shape(shape)
		if self.size != size:
			msg = f'Expr {expr} has 1D size {size}, which is incompatible with the expr socket (size {self.size})'
			raise ValueError(msg)

		return mathtype, size

	def _to_raw_value(self, expr: spux.SympyExpr, force_complex: bool = False):
		"""Cast the given expression to the appropriate raw value, with scaling guided by `self.unit`."""
		if self.unit is not None:
			pyvalue = spux.sympy_to_python(spux.scale_to_unit(expr, self.unit))
		else:
			pyvalue = spux.sympy_to_python(expr)

		# Cast complex -> tuple[float, float]
		if isinstance(pyvalue, complex) or (
			isinstance(pyvalue, int | float) and force_complex
		):
			return (pyvalue.real, pyvalue.imag)
		if isinstance(pyvalue, tuple) and all(
			isinstance(v, complex)
			or (isinstance(pyvalue, int | float) and force_complex)
			for v in pyvalue
		):
			return tuple([(v.real, v.imag) for v in pyvalue])

		return pyvalue

	def _parse_expr_str(self, expr_spstr: str) -> spux.SympyExpr | None:
		"""Parse an expression string by choosing opinionated options for `sp.sympify`.

		Uses `self._parse_expr_info()` to validate the parsed result.

		Returns:
			The parsed expression, if it manages to validate; else None.
		"""
		expr = sp.sympify(
			expr_spstr,
			locals={sym.name: sym for sym in self.symbols},
			strict=False,
			convert_xor=True,
		).subs(spux.UNIT_BY_SYMBOL)

		# Try Parsing and Returning the Expression
		try:
			self._parse_expr_info(expr)
		except ValueError:
			log.exception(
				'Couldn\'t parse expression "%s" in Expr socket.',
				expr_spstr,
			)
		else:
			return expr

		return None

	####################
	# - FlowKind: Value
	####################
	@property
	def value(self) -> spux.SympyExpr:
		"""Return the expression defined by the socket as `FlowKind.Value`.

		- **Num Dims**: Determine which property dimensionality to pull data from.
		- **MathType**: Determine which property type to pull data from.

		When `self.mathtype` is `None`, the expression is parsed from the string `self.raw_value_spstr`.

		Notes:
			Called to compute the internal `FlowKind.Value` of this socket.

		Return:
			The expression defined by the socket, in the socket's unit.

			When the string expression `self.raw_value_spstr` fails to parse,the property returns `FlowPending`.
		"""
		if self.symbols:
			expr = self.raw_value_sp
			if expr is None:
				return ct.FlowSignal.FlowPending
			return expr * (self.unit if self.unit is not None else 1)

		# Vec4 -> FlowPending
		## -> ExprSocket doesn't support Vec4 (yet?).
		## -> I mean, have you _seen_ that mess of attributes up top?
		NS = spux.NumberSize1D
		if self.size == NS.Vec4:
			return ct.Flow

		MT_Z = spux.MathType.Integer
		MT_Q = spux.MathType.Rational
		MT_R = spux.MathType.Real
		MT_C = spux.MathType.Complex
		Z = sp.Integer
		Q = sp.Rational
		R = sp.RealNumber
		return {
			NS.Scalar: {
				MT_Z: lambda: Z(self.raw_value_int),
				MT_Q: lambda: Q(self.raw_value_rat[0], self.raw_value_rat[1]),
				MT_R: lambda: R(self.raw_value_float),
				MT_C: lambda: (
					self.raw_value_complex[0] + sp.I * self.raw_value_complex[1]
				),
			},
			NS.Vec2: {
				MT_Z: lambda: sp.ImmutableMatrix([Z(i) for i in self.raw_value_int2]),
				MT_Q: lambda: sp.ImmutableMatrix(
					[Q(q[0], q[1]) for q in self.raw_value_rat2]
				),
				MT_R: lambda: sp.ImmutableMatrix([R(r) for r in self.raw_value_float2]),
				MT_C: lambda: sp.ImmutableMatrix(
					[c[0] + sp.I * c[1] for c in self.raw_value_complex2]
				),
			},
			NS.Vec3: {
				MT_Z: lambda: sp.ImmutableMatrix([Z(i) for i in self.raw_value_int3]),
				MT_Q: lambda: sp.ImmutableMatrix(
					[Q(q[0], q[1]) for q in self.raw_value_rat3]
				),
				MT_R: lambda: sp.ImmutableMatrix([R(r) for r in self.raw_value_float3]),
				MT_C: lambda: sp.ImmutableMatrix(
					[c[0] + sp.I * c[1] for c in self.raw_value_complex3]
				),
			},
		}[self.size][self.mathtype]() * (self.unit if self.unit is not None else 1)

	@value.setter
	def value(self, expr: spux.SympyExpr) -> None:
		"""Set the expression defined by the socket to a compatible `expr`.

		Notes:
			Called to set the internal `FlowKind.Value` of this socket.
		"""
		_mathtype, _size = self._parse_expr_info(expr)
		if self.symbols:
			self.raw_value_spstr = sp.sstr(expr)
		else:
			NS = spux.NumberSize1D
			MT = spux.MathType
			match (self.size, self.mathtype):
				case (NS.Scalar, MT.Integer):
					self.raw_value_int = self._to_raw_value(expr)
				case (NS.Scalar, MT.Rational):
					self.raw_value_rat = self._to_raw_value(expr)
				case (NS.Scalar, MT.Real):
					self.raw_value_float = self._to_raw_value(expr)
				case (NS.Scalar, MT.Complex):
					self.raw_value_complex = self._to_raw_value(
						expr, force_complex=True
					)

				case (NS.Vec2, MT.Integer):
					self.raw_value_int2 = self._to_raw_value(expr)
				case (NS.Vec2, MT.Rational):
					self.raw_value_rat2 = self._to_raw_value(expr)
				case (NS.Vec2, MT.Real):
					self.raw_value_float2 = self._to_raw_value(expr)
				case (NS.Vec2, MT.Complex):
					self.raw_value_complex2 = self._to_raw_value(
						expr, force_complex=True
					)

				case (NS.Vec3, MT.Integer):
					self.raw_value_int3 = self._to_raw_value(expr)
				case (NS.Vec3, MT.Rational):
					self.raw_value_rat3 = self._to_raw_value(expr)
				case (NS.Vec3, MT.Real):
					self.raw_value_float3 = self._to_raw_value(expr)
				case (NS.Vec3, MT.Complex):
					self.raw_value_complex3 = self._to_raw_value(
						expr, force_complex=True
					)

	####################
	# - FlowKind: Range
	####################
	@property
	def lazy_range(self) -> ct.RangeFlow:
		"""Return the not-yet-computed uniform array defined by the socket.

		Notes:
			Called to compute the internal `FlowKind.Range` of this socket.

		Return:
			The range of lengths, which uses no symbols.
		"""
		if self.symbols:
			return ct.RangeFlow(
				start=self.raw_min_sp,
				stop=self.raw_max_sp,
				steps=self.steps,
				scaling=self.scaling,
				unit=self.unit,
				symbols=self.symbols,
			)

		MT_Z = spux.MathType.Integer
		MT_Q = spux.MathType.Rational
		MT_R = spux.MathType.Real
		MT_C = spux.MathType.Complex
		Z = sp.Integer
		Q = sp.Rational
		R = sp.RealNumber

		min_bound, max_bound = {
			MT_Z: lambda: [Z(bound) for bound in self.raw_range_int],
			MT_Q: lambda: [Q(bound[0], bound[1]) for bound in self.raw_range_rat],
			MT_R: lambda: [R(bound) for bound in self.raw_range_float],
			MT_C: lambda: [
				bound[0] + sp.I * bound[1] for bound in self.raw_range_complex
			],
		}[self.mathtype]()

		return ct.RangeFlow(
			start=min_bound,
			stop=max_bound,
			steps=self.steps,
			scaling=self.scaling,
			unit=self.unit,
		)

	@lazy_range.setter
	def lazy_range(self, value: ct.RangeFlow) -> None:
		"""Set the not-yet-computed uniform array defined by the socket.

		Notes:
			Called to compute the internal `FlowKind.Range` of this socket.
		"""
		self.steps = value.steps
		self.scaling = value.scaling

		if self.symbols:
			self.raw_min_spstr = sp.sstr(value.start)
			self.raw_max_spstr = sp.sstr(value.stop)

		else:
			MT_Z = spux.MathType.Integer
			MT_Q = spux.MathType.Rational
			MT_R = spux.MathType.Real
			MT_C = spux.MathType.Complex

			unit = value.unit if value.unit is not None else 1
			if self.mathtype == MT_Z:
				self.raw_range_int = [
					self._to_raw_value(bound * unit)
					for bound in [value.start, value.stop]
				]
			elif self.mathtype == MT_Q:
				self.raw_range_rat = [
					self._to_raw_value(bound * unit)
					for bound in [value.start, value.stop]
				]
			elif self.mathtype == MT_R:
				self.raw_range_float = [
					self._to_raw_value(bound * unit)
					for bound in [value.start, value.stop]
				]
			elif self.mathtype == MT_C:
				self.raw_range_complex = [
					self._to_raw_value(bound * unit, force_complex=True)
					for bound in [value.start, value.stop]
				]

	####################
	# - FlowKind: Func (w/Params if Constant)
	####################
	@property
	def lazy_func(self) -> ct.FuncFlow:
		"""Returns a lazy value that computes the expression returned by `self.value`.

		If `self.value` has unknown symbols (as indicated by `self.symbols`), then these will be the arguments of the `FuncFlow`.
		Otherwise, the returned lazy value function will be a simple excuse for `self.params` to pass the verbatim `self.value`.
		"""
		# Symbolic
		## -> `self.value` is guaranteed to be an expression with unknowns.
		## -> The function computes `self.value` with unknowns as arguments.
		if self.symbols:
			return ct.FuncFlow(
				func=sp.lambdify(
					self.sorted_symbols,
					spux.scale_to_unit(self.value, self.unit),
					'jax',
				),
				func_args=[spux.MathType.from_expr(sym) for sym in self.sorted_symbols],
				supports_jax=True,
			)

		# Constant
		## -> When a `self.value` has no unknowns, use a dummy function.
		## -> ("Dummy" as in returns the same argument that it takes).
		## -> This is an excuse to let `ParamsFlow` pass `self.value` verbatim.
		## -> Generally only useful for operations with other expressions.
		return ct.FuncFlow(
			func=lambda v: v,
			func_args=[
				self.physical_type if self.physical_type is not None else self.mathtype
			],
			supports_jax=True,
		)

	@property
	def params(self) -> ct.ParamsFlow:
		"""Returns parameter symbols/values to accompany `self.lazy_func`.

		If `self.value` has unknown symbols (as indicated by `self.symbols`), then these will be passed into `ParamsFlow`, which will thus be parameterized (and require realization before use).
		Otherwise, `self.value` is passed verbatim as the only `ParamsFlow.func_arg`.
		"""
		# Symbolic
		## -> The Expr socket does not declare actual values for the symbols.
		## -> They should be realized later, ex. in a Viz node.
		## -> Therefore, we just dump the symbols. Easy!
		## -> NOTE: func_args must have the same symbol order as was lambdified.
		if self.symbols:
			return ct.ParamsFlow(
				func_args=self.sorted_symbols,
				symbols=self.symbols,
			)

		# Constant
		## -> Simply pass self.value verbatim as a function argument.
		## -> Easy dice, easy life!
		return ct.ParamsFlow(func_args=[self.value])

	@property
	def info(self) -> ct.ArrayFlow:
		r"""Returns parameter symbols/values to accompany `self.lazy_func`.

		The output name/size/mathtype/unit corresponds directly the `ExprSocket`.

		If `self.symbols` has entries, then these will propagate as dimensions with unresolvable `RangeFlow` index descriptions.
		The index range will be $(-\infty,\infty)$, with $0$ steps and no unit.
		The order/naming matches `self.params` and `self.lazy_func`.

		Otherwise, only the output name/size/mathtype/unit corresponding to the socket is passed along.
		"""
		output_sim_sym = (
			sim_symbols.SimSymbol(
				sym_name=self.output_name,
				mathtype=self.mathtype,
				physical_type=self.physical_type,
				unit=self.unit,
				rows=self.size.rows,
				cols=self.size.cols,
			),
		)
		if self.symbols:
			return ct.InfoFlow(
				dims={sim_sym: None for sim_sym in self.active_symbols},
				output=output_sim_sym,
			)

		# Constant
		return ct.InfoFlow(output=output_sim_sym)

	####################
	# - FlowKind: Capabilities
	####################
	@property
	def capabilities(self) -> None:
		return ct.CapabilitiesFlow(
			socket_type=self.socket_type,
			active_kind=self.active_kind,
		)

	####################
	# - UI: Label Row
	####################
	def draw_label_row(self, row: bpy.types.UILayout, text) -> None:
		"""Draw the unlinked input label row, with a unit dropdown (if `self.active_unit`)."""
		# Has Unit: Draw Label and Unit Dropdown
		if self.active_unit is not None:
			split = row.split(factor=0.6, align=True)

			_row = split.row(align=True)
			_row.label(text=text)

			_col = split.column(align=True)
			_col.prop(self, self.blfields['active_unit'], text='')

		# No Unit: Draw Label
		else:
			row.label(text=text)

	def draw_input_label_row(self, row: bpy.types.UILayout, text) -> None:
		"""Provide a dropdown for enabling the `InfoFlow` UI in the linked input label row.

		Notes:
			Whether information about the expression passing through a linked socket is shown is governed by `self.show_info_columns`.
		"""
		info = self.compute_data(kind=ct.FlowKind.Info)
		has_info = not ct.FlowSignal.check(info)

		if has_info:
			split = row.split(factor=0.85, align=True)
			_row = split.row(align=False)
		else:
			_row = row

		_row.label(text=text)
		if has_info:
			if self.show_info_columns:
				_row.prop(self, self.blfields['info_columns'])

			_row = split.row(align=True)
			_row.alignment = 'RIGHT'
			_row.prop(
				self,
				self.blfields['show_info_columns'],
				toggle=True,
				text='',
				icon=ct.Icon.ToggleSocketInfo,
			)

	def draw_output_label_row(self, row: bpy.types.UILayout, text) -> None:
		"""Provide a dropdown for enabling the `InfoFlow` UI in the linked output label row.

		Extremely similar to `draw_input_label_row`, except for some tricky right-alignment.

		Notes:
			Whether information about the expression passing through a linked socket is shown is governed by `self.show_info_columns`.
		"""
		info = self.compute_data(kind=ct.FlowKind.Info)
		has_info = not ct.FlowSignal.check(info)

		if has_info:
			split = row.split(factor=0.15, align=True)

			_row = split.row(align=True)
			_row.prop(
				self,
				self.blfields['show_info_columns'],
				toggle=True,
				text='',
				icon=ct.Icon.ToggleSocketInfo,
			)

			_row = split.row(align=False)
			_row.alignment = 'RIGHT'
			if self.show_info_columns:
				_row.prop(self, self.blfields['info_columns'])
			else:
				_col = _row.column()
				_col.alignment = 'EXPAND'
				_col.label(text='')
		else:
			_row = row

		_row.label(text=text)

	####################
	# - UI: Active FlowKind
	####################
	def draw_value(self, col: bpy.types.UILayout) -> None:
		"""Draw the socket body for a single value/expression.

		This implements the base UI for `ExprSocket`, for when `self.size`, `self.mathtype`, `self.physical_type`, and `self.symbols` are set.

		Notes:
			Drawn when `self.active_kind == FlowKind.Value`.

			Alone, `draw_value` provides no mechanism for altering expression constraints like size.
			Thus, `FlowKind.Value` is a good choice for when the expression must be of a very particular type.

			However, `draw_value` may also be called by the `draw_*` methods of other `FlowKinds`, who may choose to layer more flexibility around this base UI.
		"""
		if self.symbols:
			col.prop(self, self.blfields['raw_value_spstr'], text='')

		else:
			NS = spux.NumberSize1D
			MT = spux.MathType
			match (self.size, self.mathtype):
				case (NS.Scalar, MT.Integer):
					col.prop(self, self.blfields['raw_value_int'], text='')
				case (NS.Scalar, MT.Rational):
					col.prop(self, self.blfields['raw_value_rat'], text='')
				case (NS.Scalar, MT.Real):
					col.prop(self, self.blfields['raw_value_float'], text='')
				case (NS.Scalar, MT.Complex):
					col.prop(self, self.blfields['raw_value_complex'], text='')

				case (NS.Vec2, MT.Integer):
					col.prop(self, self.blfields['raw_value_int2'], text='')
				case (NS.Vec2, MT.Rational):
					col.prop(self, self.blfields['raw_value_rat2'], text='')
				case (NS.Vec2, MT.Real):
					col.prop(self, self.blfields['raw_value_float2'], text='')
				case (NS.Vec2, MT.Complex):
					col.prop(self, self.blfields['raw_value_complex2'], text='')

				case (NS.Vec3, MT.Integer):
					col.prop(self, self.blfields['raw_value_int3'], text='')
				case (NS.Vec3, MT.Rational):
					col.prop(self, self.blfields['raw_value_rat3'], text='')
				case (NS.Vec3, MT.Real):
					col.prop(self, self.blfields['raw_value_float3'], text='')
				case (NS.Vec3, MT.Complex):
					col.prop(self, self.blfields['raw_value_complex3'], text='')

		# Symbol Information
		if self.symbols:
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

	def draw_lazy_range(self, col: bpy.types.UILayout) -> None:
		"""Draw the socket body for a simple, uniform range of values between two values/expressions.

		Drawn when `self.active_kind == FlowKind.Range`.

		Notes:
			If `self.steps == 0`, then the `Range` is considered to have a to-be-determined number of steps.
			As such, `self.steps` won't be exposed in the UI.
		"""
		if self.symbols:
			col.prop(self, self.blfields['raw_min_spstr'], text='')
			col.prop(self, self.blfields['raw_max_spstr'], text='')

		else:
			MT_Z = spux.MathType.Integer
			MT_Q = spux.MathType.Rational
			MT_R = spux.MathType.Real
			MT_C = spux.MathType.Complex
			if self.mathtype == MT_Z:
				col.prop(self, self.blfields['raw_range_int'], text='')
			elif self.mathtype == MT_Q:
				col.prop(self, self.blfields['raw_range_rat'], text='')
			elif self.mathtype == MT_R:
				col.prop(self, self.blfields['raw_range_float'], text='')
			elif self.mathtype == MT_C:
				col.prop(self, self.blfields['raw_range_complex'], text='')

		if self.steps != 0:
			col.prop(self, self.blfields['steps'], text='')

	def draw_lazy_func(self, col: bpy.types.UILayout) -> None:
		"""Draw the socket body for a single flexible value/expression, for down-chain lazy evaluation.

		This implements the most flexible variant of the `ExprSocket` UI, providing the user with full runtime-configuration of the exact `self.size`, `self.mathtype`, `self.physical_type`, and `self.symbols` of the expression.

		Notes:
			Drawn when `self.active_kind == FlowKind.Func`.

			This is an ideal choice for ex. math nodes that need to accept arbitrary expressions as inputs, with an eye towards lazy evaluation of ex. symbolic terms.

			Uses `draw_value` to draw the base UI
		"""
		if self.show_func_ui:
			# Output Name Selector
			## -> The name of the output
			col.prop(self, self.blfields['output_name'], text='')

			# Physical Type Selector
			## -> Determines whether/which unit-dropdown will be shown.
			col.prop(self, self.blfields['physical_type'], text='')

			# Non-Symbolic: Size/Mathtype Selector
			## -> Symbols imply str expr input.
			## -> For arbitrary str exprs, size/mathtype are derived from the expr.
			## -> Otherwise, size/mathtype must be pre-specified for a nice UI.
			if not self.symbols:
				row = col.row(align=True)
				row.prop(self, self.blfields['size'], text='')
				row.prop(self, self.blfields['mathtype'], text='')

			# Base UI
			## -> Draws the UI appropriate for the above choice of constraints.
			self.draw_value(col)

			# Symbol UI
			## -> Draws the UI appropriate for the above choice of constraints.
			## -> TODO

	####################
	# - UI: InfoFlow
	####################
	def draw_info(self, info: ct.InfoFlow, col: bpy.types.UILayout) -> None:
		if self.active_kind == ct.FlowKind.Func and self.show_info_columns:
			row = col.row()
			box = row.box()
			grid = box.grid_flow(
				columns=len(self.info_columns) + 1,
				row_major=True,
				even_columns=True,
				# even_rows=True,
				align=True,
			)

			# Dimensions
			for dim in info.dims:
				dim_idx = info.dims[dim]
				grid.label(text=dim.name_pretty)
				if InfoDisplayCol.Length in self.info_columns:
					grid.label(text=str(len(dim_idx)))
				if InfoDisplayCol.MathType in self.info_columns:
					grid.label(text=spux.MathType.to_str(dim_idx.mathtype))
				if InfoDisplayCol.Unit in self.info_columns:
					grid.label(text=spux.sp_to_str(dim_idx.unit))

			# Outputs
			grid.label(text=info.output.name_pretty)
			if InfoDisplayCol.Length in self.info_columns:
				grid.label(text='', icon=ct.Icon.DataSocketOutput)
			if InfoDisplayCol.MathType in self.info_columns:
				grid.label(
					text=(
						spux.MathType.to_str(info.output.mathtype)
						+ (
							'ˣ'.join(
								[
									unicode_superscript(out_axis)
									for out_axis in info.output.shape
								]
							)
							if info.output.shape
							else ''
						)
					)
				)
			if InfoDisplayCol.Unit in self.info_columns:
				grid.label(text=f'{spux.sp_to_str(info.output.unit)}')


####################
# - Socket Configuration
####################
class ExprSocketDef(base.SocketDef):
	socket_type: ct.SocketType = ct.SocketType.Expr
	active_kind: typ.Literal[
		ct.FlowKind.Value,
		ct.FlowKind.Range,
		ct.FlowKind.Array,
		ct.FlowKind.Func,
	] = ct.FlowKind.Value
	output_name: sim_symbols.SimSymbolName = sim_symbols.SimSymbolName

	# Socket Interface
	size: spux.NumberSize1D = spux.NumberSize1D.Scalar
	mathtype: spux.MathType = spux.MathType.Real
	physical_type: spux.PhysicalType = spux.PhysicalType.NonPhysical

	default_unit: spux.Unit | None = None
	default_symbols: list[sim_symbols.SimSymbol] = pyd.Field(default_factory=list)

	# FlowKind: Value
	default_value: spux.SympyExpr = 0
	abs_min: spux.SympyExpr | None = None
	abs_max: spux.SympyExpr | None = None

	# FlowKind: Range
	default_min: spux.SympyExpr = 0
	default_max: spux.SympyExpr = 1
	default_steps: int = 2
	default_scaling: ct.ScalingMode = ct.ScalingMode.Lin

	# UI
	show_func_ui: bool = True
	show_info_columns: bool = False

	####################
	# - Parse Unit and/or Physical Type
	####################
	@pyd.model_validator(mode='after')
	def parse_default_unit(self) -> typ.Self:
		"""Guarantees that a valid default unit is defined, with respect to a given `self.physical_type`.

		If no `self.default_unit` is given, then the physical type's builtin default unit is inserted.
		"""
		if (
			self.physical_type is not spux.PhysicalType.NonPhysical
			and self.default_unit is None
		):
			self.default_unit = self.physical_type.default_unit

		return self

	@pyd.model_validator(mode='after')
	def parse_physical_type_from_unit(self) -> typ.Self:
		"""Guarantees that a valid physical type is defined based on the unit.

		If no `self.physical_type` is given, but a unit is defined, then `spux.PhysicalType.from_unit()` is used to find an appropriate PhysicalType.

		Raises:
			ValueError: If `self.default_unit` has no obvious physical type.
				This might happen if `self.default_unit` isn't a unit at all!
		"""
		if (
			self.physical_type is spux.PhysicalType.NonPhysical
			and self.default_unit is not None
		):
			physical_type = spux.PhysicalType.from_unit(self.default_unit)
			if physical_type is spux.PhysicalType.NonPhysical:
				msg = f'ExprSocket: Defined unit {self.default_unit} has no obvious physical type defined for it.'
				raise ValueError(msg)

			self.physical_type = physical_type
		return self

	@pyd.model_validator(mode='after')
	def assert_physical_type_mathtype_compatibility(self) -> typ.Self:
		"""Guarantees that the physical type is compatible with `self.mathtype`.

		The `self.physical_type.valid_mathtypes` method is used to perform this check.

		Raises:
			ValueError: If `self.default_unit` has no obvious physical type.
				This might happen if `self.default_unit` isn't a unit at all!
		"""
		# Check MathType-PhysicalType Compatibility
		## -> NOTE: NonPhysical has a valid_mathtypes list.
		if self.mathtype not in self.physical_type.valid_mathtypes:
			msg = f'ExprSocket: Defined unit {self.default_unit} has no obvious physical type defined for it.'
			raise ValueError(msg)

		return self

	@pyd.model_validator(mode='after')
	def assert_unit_is_valid_in_physical_type(self) -> str:
		"""Guarantees that the given unit is a valid unit within the given `spux.PhysicalType`.

		This is implemented by checking `self.physical_type.valid_units`.

		Raises:
			ValueError: If `self.default_unit` has no obvious physical type.
				This might happen if `self.default_unit` isn't a unit at all!
		"""
		if (
			self.default_unit is not None
			and self.default_unit not in self.physical_type.valid_units
		):
			msg = f'ExprSocket: Defined unit {self.default_unit} is not a valid unit of {self.physical_type} (valid units = {self.physical_type.valid_units})'
			raise ValueError(msg)

		return self

	####################
	# - Parse FlowKind.Value
	####################
	@pyd.model_validator(mode='after')
	def parse_default_value_size(self) -> typ.Self:
		"""Guarantees that the default value is correctly shaped.

		If a single number for `self.default_value` is given, then it will be broadcast into the given `self.size.shape`.

		Raises:
			ValueError: If `self.default_value` is shaped, but with a shape not identical to `self.size`.
		"""
		# Default Value is sp.Matrix
		## -> Let the user take responsibility for shape
		if isinstance(self.default_value, sp.MatrixBase):
			if self.size.supports_shape(self.default_value.shape):
				return self

			msg = f"ExprSocket: Default value {self.default_value} is shaped, but its shape {self.default_value.shape} doesn't match the shape of the ExprSocket {self.size.shape}"
			raise ValueError(msg)

		if self.size.shape is not None:
			# Coerce Number -> Column 0-Vector
			## -> TODO: We don't strictly know if default_value is a number.
			if len(self.size.shape) == 1:
				self.default_value = self.default_value * sp.Matrix.ones(
					self.size.shape[0], 1
				)

			# Coerce Number -> 0-Matrix
			## -> TODO: We don't strictly know if default_value is a number.
			if len(self.size.shape) > 1:
				self.default_value = self.default_value * sp.Matrix.ones(
					*self.size.shape
				)

		return self

	@pyd.model_validator(mode='after')
	def parse_default_value_number(self) -> typ.Self:
		"""Guarantees that the default value is a sympy expression w/valid (possibly pre-coerced) MathType.

		If `self.default_value` is a scalar Python type, it will be coerced into the corresponding Sympy type using `sp.S`, after coersion to the correct Python type using `self.mathtype.coerce_compatible_pyobj()`.

		Raises:
			ValueError: If `self.default_value` has no obvious, coerceable `spux.MathType` compatible with `self.mathtype`, as determined by `spux.MathType.has_mathtype`.
		"""
		mathtype_guide = spux.MathType.has_mathtype(self.default_value)

		# None: No Obvious Mathtype
		if mathtype_guide is None:
			msg = f'ExprSocket: Type of default value {self.default_value} (type {type(self.default_value)})'
			raise ValueError(msg)

		# PyType: Coerce from PyType
		if mathtype_guide == 'pytype':
			dv_mathtype = spux.MathType.from_pytype(type(self.default_value))
			if self.mathtype.is_compatible(dv_mathtype):
				self.default_value = sp.S(
					self.mathtype.coerce_compatible_pyobj(self.default_value)
				)
			else:
				msg = f'ExprSocket: Mathtype {dv_mathtype} of default value {self.default_value} (type {type(self.default_value)}) is incompatible with socket MathType {self.mathtype}'
				raise ValueError(msg)

		# Expr: Merely Check MathType Compatibility
		if mathtype_guide == 'expr':
			dv_mathtype = spux.MathType.from_expr(self.default_value)
			if not self.mathtype.is_compatible(dv_mathtype):
				msg = f'ExprSocket: Mathtype {dv_mathtype} of default value expression {self.default_value} (type {type(self.default_value)}) is incompatible with socket MathType {self.mathtype}'
				raise ValueError(msg)

		return self

	####################
	# - Parse FlowKind.Range
	####################
	@pyd.field_validator('default_steps')
	@classmethod
	def steps_must_be_0_or_gte_2(cls, v: int) -> int:
		r"""Checks that steps is either 0 (not currently set), or $\ge 2$."""
		if not (v >= 2 or v == 0):  # noqa: PLR2004
			msg = f'Default steps {v} must either be greater than or equal to 2, or 0 (denoting that no steps are currently given)'
			raise ValueError(msg)

		return v

	@pyd.model_validator(mode='after')
	def parse_default_lazy_range_numbers(self) -> typ.Self:
		"""Guarantees that the default `ct.Range` bounds are sympy expressions.

		If `self.default_value` is a scalar Python type, it will be coerced into the corresponding Sympy type using `sp.S`.

		Raises:
			ValueError: If `self.default_value` has no obvious `spux.MathType`, as determined by `spux.MathType.has_mathtype`.
		"""
		new_bounds = [None, None]
		for i, bound in enumerate([self.default_min, self.default_max]):
			mathtype_guide = spux.MathType.has_mathtype(bound)

			# None: No Obvious Mathtype
			if mathtype_guide is None:
				msg = f'ExprSocket: A default bound {bound} (type {type(bound)}) has no MathType.'
				raise ValueError(msg)

			# PyType: Coerce from PyType
			if mathtype_guide == 'pytype':
				dv_mathtype = spux.MathType.from_pytype(type(bound))
				if self.mathtype.is_compatible(dv_mathtype):
					new_bounds[i] = sp.S(self.mathtype.coerce_compatible_pyobj(bound))
				else:
					msg = f'ExprSocket: Mathtype {dv_mathtype} of a bound {bound} (type {type(bound)}) is incompatible with socket MathType {self.mathtype}'
					raise ValueError(msg)

			# Expr: Merely Check MathType Compatibility
			if mathtype_guide == 'expr':
				dv_mathtype = spux.MathType.from_expr(bound)
				if not self.mathtype.is_compatible(dv_mathtype):
					msg = f'ExprSocket: Mathtype {dv_mathtype} of a default Range min or max expression {bound} (type {type(self.default_value)}) is incompatible with socket MathType {self.mathtype}'
					raise ValueError(msg)

			# Coerce from Infinite
			if bound.is_infinite and self.mathtype is spux.MathType.Integer:
				new_bounds[i] = sp.S(-1) if i == 0 else sp.S(1)
			if bound.is_infinite and self.mathtype is spux.MathType.Rational:
				new_bounds[i] = sp.Rational(-1, 1) if i == 0 else sp.Rational(1, 1)
			if bound.is_infinite and self.mathtype is spux.MathType.Real:
				new_bounds[i] = sp.S(-1.0) if i == 0 else sp.S(1.0)

		if new_bounds[0] is not None:
			self.default_min = new_bounds[0]
		if new_bounds[1] is not None:
			self.default_max = new_bounds[1]

		return self

	@pyd.model_validator(mode='after')
	def parse_default_lazy_range_size(self) -> typ.Self:
		"""Guarantees that the default `ct.Range` bounds are unshaped.

		Raises:
			ValueError: If `self.default_min` or `self.default_max` are shaped.
		"""
		# Check ActiveKind and Size
		## -> NOTE: This doesn't protect against dynamic changes to either.
		if (
			self.active_kind == ct.FlowKind.Range
			and self.size is not spux.NumberSize1D.Scalar
		):
			msg = "Can't have a non-Scalar size when Range is set as the active kind."
			raise ValueError(msg)

		# Check that Bounds are Shapeless
		for bound in [self.default_min, self.default_max]:
			if hasattr(bound, 'shape'):
				msg = f'ExprSocket: A default bound {bound} (type {type(bound)}) has a shape, but Range supports no shape in ExprSockets.'
				raise ValueError(msg)

		return self

	####################
	# - Validators - Assertion
	####################
	@pyd.model_validator(mode='after')
	def symbols_value(self) -> typ.Self:
		if (
			self.default_value.free_symbols
			and not self.default_value.free_symbols.issubset(self.symbols)
		):
			msg = f'Tried to set default value {self.default_value} with free symbols {self.default_value.free_symbols}, which is incompatible with socket symbols {self.symbols}'
			raise ValueError(msg)

		return self

	@pyd.model_validator(mode='after')
	def shape_value(self) -> typ.Self:
		shape = spux.parse_shape(self.default_value)
		if not self.size.supports_shape(shape):
			msg = f'Default expr {self.default_value} has non-1D shape {shape}, which is incompatible with the expr socket def (size {self.size})'
			raise ValueError(msg)

		size = spux.NumberSize1D.from_shape(shape)
		if self.size != size:
			msg = f'Default expr size {size} is incompatible with the expr socket (size {self.size})'
			raise ValueError(msg)

		return self

	####################
	# - Initialization
	####################
	def init(self, bl_socket: ExprBLSocket) -> None:
		bl_socket.active_kind = self.active_kind
		bl_socket.output_name = self.output_name

		# Socket Interface
		## -> Recall that auto-updates are turned off during init()
		bl_socket.size = self.size
		bl_socket.mathtype = self.mathtype
		bl_socket.physical_type = self.physical_type
		bl_socket.active_symbols = self.symbols

		# FlowKind.Value
		## -> We must take units into account when setting bl_socket.value
		if self.physical_type is not spux.PhysicalType.NonPhysical:
			bl_socket.active_unit = sp.sstr(self.default_unit)
			bl_socket.value = self.default_value * self.default_unit
		else:
			bl_socket.value = self.default_value

		bl_socket.prev_unit = bl_socket.active_unit

		# FlowKind.Range
		## -> We can directly pass None to unit.
		bl_socket.lazy_range = ct.RangeFlow(
			start=self.default_min,
			stop=self.default_max,
			steps=self.default_steps,
			scaling=self.default_scaling,
			unit=self.default_unit,
		)

		# UI
		bl_socket.show_func_ui = self.show_func_ui
		bl_socket.show_info_columns = self.show_info_columns

		# Info Draw
		bl_socket.use_info_draw = True


####################
# - Blender Registration
####################
BL_REGISTER = [
	ExprBLSocket,
]
