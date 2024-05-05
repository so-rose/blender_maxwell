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

import enum
import typing as typ

import bpy
import pydantic as pyd
import sympy as sp

from blender_maxwell.utils import bl_cache, logger
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


def unicode_superscript(n):
	return ''.join(['⁰¹²³⁴⁵⁶⁷⁸⁹'[ord(c) - ord('0')] for c in str(n)])


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


class ExprBLSocket(base.MaxwellSimSocket):
	"""The `Expr` ("Expression") socket is an accessible approach to specifying any expression.

	- **Shape**: There is an intuitive UI for scalar, 2D, and 3D, but the `Expr` socket also supports parsing mathematical expressions of any shape (including matrices).
	- **Math Type**: Support integer, rational, real, and complex mathematical types, for which there is an intuitive UI for scalar, 2D, and 3D cases.
	- **Physical Type**: Supports the declaration of a physical unit dimension, for which a UI exists for the user to switch between long lists of valid units for that dimension, with automatic conversion of the value. This causes the expression to become unit-aware, which will be respected when using it for math.
	- **Symbols**: Supports the use of variables (each w/predefined `MathType`) to define arbitrary mathematical expressions, which can be used as part of a function composition chain and/or as a parameter realized at `Viz` / when generating batched simulations / when performing gradient-based optimization.
	- **Information UX**: All information encoded by the expression is presented using an intuitive UI, including filterable access to the shape of any data passing through a linked socket.
	"""

	socket_type = ct.SocketType.Expr
	bl_label = 'Expr'
	use_info_draw = True

	####################
	# - Properties
	####################
	shape: tuple[int, ...] | None = bl_cache.BLField(None)
	mathtype: spux.MathType = bl_cache.BLField(spux.MathType.Real, prop_ui=True)
	physical_type: spux.PhysicalType | None = bl_cache.BLField(None)
	symbols: frozenset[sp.Symbol] = bl_cache.BLField(frozenset())

	active_unit: enum.Enum = bl_cache.BLField(
		None, enum_cb=lambda self, _: self.search_units(), prop_ui=True
	)

	# UI: Value
	## Expression
	raw_value_spstr: str = bl_cache.BLField('', prop_ui=True)
	## 1D
	raw_value_int: int = bl_cache.BLField(0, prop_ui=True)
	raw_value_rat: Int2 = bl_cache.BLField((0, 1), prop_ui=True)
	raw_value_float: float = bl_cache.BLField(0.0, float_prec=4, prop_ui=True)
	raw_value_complex: Float2 = bl_cache.BLField((0.0, 0.0), float_prec=4, prop_ui=True)
	## 2D
	raw_value_int2: Int2 = bl_cache.BLField((0, 0), prop_ui=True)
	raw_value_rat2: Int22 = bl_cache.BLField(((0, 1), (0, 1)), prop_ui=True)
	raw_value_float2: Float2 = bl_cache.BLField((0.0, 0.0), float_prec=4, prop_ui=True)
	raw_value_complex2: Float22 = bl_cache.BLField(
		((0.0, 0.0), (0.0, 0.0)), float_prec=4, prop_ui=True
	)
	## 3D
	raw_value_int3: Int3 = bl_cache.BLField((0, 0, 0), prop_ui=True)
	raw_value_rat3: Int32 = bl_cache.BLField(((0, 1), (0, 1), (0, 1)), prop_ui=True)
	raw_value_float3: Float3 = bl_cache.BLField(
		(0.0, 0.0, 0.0), float_prec=4, prop_ui=True
	)
	raw_value_complex3: Float32 = bl_cache.BLField(
		((0.0, 0.0), (0.0, 0.0), (0.0, 0.0)), float_prec=4, prop_ui=True
	)

	# UI: LazyArrayRange
	steps: int = bl_cache.BLField(2, abs_min=2, prop_ui=True)
	## Expression
	raw_min_spstr: str = bl_cache.BLField('', prop_ui=True)
	raw_max_spstr: str = bl_cache.BLField('', prop_ui=True)
	## By MathType
	raw_range_int: Int2 = bl_cache.BLField((0, 1), prop_ui=True)
	raw_range_rat: Int22 = bl_cache.BLField(((0, 1), (1, 1)), prop_ui=True)
	raw_range_float: Float2 = bl_cache.BLField((0.0, 1.0), prop_ui=True)
	raw_range_complex: Float22 = bl_cache.BLField(
		((0.0, 0.0), (1.0, 1.0)), float_prec=4, prop_ui=True
	)

	# UI: Info
	show_info_columns: bool = bl_cache.BLField(False, prop_ui=True)
	info_columns: InfoDisplayCol = bl_cache.BLField(
		{InfoDisplayCol.MathType, InfoDisplayCol.Unit},
		prop_ui=True,
		enum_many=True,
	)

	####################
	# - Computed: Raw Expressions
	####################
	@property
	def sorted_symbols(self) -> list[sp.Symbol]:
		"""Retrieves all symbols and sorts them by name.

		Returns:
			Repeateably ordered list of symbols.
		"""
		return sorted(self.symbols, key=lambda sym: sym.name)

	@property
	def raw_value_sp(self) -> spux.SympyExpr:
		return self._parse_expr_str(self.raw_value_spstr)

	@property
	def raw_min_sp(self) -> spux.SympyExpr:
		return self._parse_expr_str(self.raw_min_spstr)

	@property
	def raw_max_sp(self) -> spux.SympyExpr:
		return self._parse_expr_str(self.raw_max_spstr)

	####################
	# - Computed: Units
	####################
	def search_units(self) -> list[ct.BLEnumElement]:
		if self.physical_type is not None:
			return [
				(sp.sstr(unit), spux.sp_to_str(unit), sp.sstr(unit), '', i)
				for i, unit in enumerate(self.physical_type.valid_units)
			]
		return []

	@bl_cache.cached_bl_property()
	def unit(self) -> spux.Unit | None:
		"""Gets the current active unit.

		Returns:
			The current active `sympy` unit.

			If the socket expression is unitless, this returns `None`.
		"""
		if self.active_unit is not None:
			return spux.unit_str_to_unit(self.active_unit)

		return None

	@unit.setter
	def unit(self, unit: spux.Unit | None) -> None:
		"""Set the unit, without touching the `raw_*` UI properties.

		Notes:
			To set a new unit, **and** convert the `raw_*` UI properties to the new unit, use `self.convert_unit()` instead.
		"""
		if self.physical_type is not None:
			if unit in self.physical_type.valid_units:
				self.active_unit = sp.sstr(unit)
			else:
				msg = f'Tried to set invalid unit {unit} (physical type "{self.physical_type}" only supports "{self.physical_type.valid_units}")'
				raise ValueError(msg)
		elif unit is not None:
			msg = f'Tried to set invalid unit {unit} (physical type is {self.physical_type}, and has no unit support!)")'
			raise ValueError(msg)

	def convert_unit(self, unit_to: spux.Unit) -> None:
		current_value = self.value
		current_lazy_array_range = self.lazy_array_range

		# Old Unit Not in Physical Type
		## -> This happens when dynamically altering self.physical_type
		if self.unit in self.physical_type.valid_units:
			self.unit = bl_cache.Signal.InvalidateCache

			self.value = current_value
			self.lazy_array_range = current_lazy_array_range
		else:
			self.unit = bl_cache.Signal.InvalidateCache

			# Workaround: Manually Jiggle FlowKind Invalidation
			self.value = self.value
			self.lazy_array_range = self.lazy_array_range

	####################
	# - Property Callback
	####################
	def on_socket_prop_changed(self, prop_name: str) -> None:
		if prop_name == 'physical_type':
			self.active_unit = bl_cache.Signal.ResetEnumItems
		if prop_name == 'active_unit' and self.active_unit is not None:
			self.convert_unit(spux.unit_str_to_unit(self.active_unit))

	####################
	# - Methods
	####################
	def _parse_expr_info(
		self, expr: spux.SympyExpr
	) -> tuple[spux.MathType, tuple[int, ...] | None, spux.UnitDimension]:
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
		if shape != self.shape and not (
			shape is not None
			and self.shape is not None
			and len(self.shape) == 1
			and 1 in shape
		):
			msg = f'Expr {expr} has shape {shape}, which is incompatible with the expr socket (shape {self.shape})'
			raise ValueError(msg)

		return mathtype, shape

	def _to_raw_value(self, expr: spux.SympyExpr, force_complex: bool = False):
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

	def _parse_expr_str(self, expr_spstr: str) -> None:
		expr = sp.sympify(
			expr_spstr,
			locals={sym.name: sym for sym in self.symbols},
			strict=False,
			convert_xor=True,
		).subs(spux.UNIT_BY_SYMBOL) * (self.unit if self.unit is not None else 1)

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
		"""Return the expression defined by the socket.

		- **Num Dims**: Determine which property dimensionality to pull data from.
		- **MathType**: Determine which property type to pull data from.

		When `self.mathtype` is `None`, the expression is parsed from the string `self.raw_value_spstr`.

		Notes:
			Called to compute the internal `FlowKind.Value` of this socket.

		Return:
			The expression defined by the socket, in the socket's unit.
		"""
		if self.symbols or self.shape not in [None, (2,), (3,)]:
			expr = self.raw_value_sp
			if expr is None:
				return ct.FlowSignal.FlowPending
			return expr

		MT_Z = spux.MathType.Integer
		MT_Q = spux.MathType.Rational
		MT_R = spux.MathType.Real
		MT_C = spux.MathType.Complex
		Z = sp.Integer
		Q = sp.Rational
		R = sp.RealNumber
		return {
			None: {
				MT_Z: lambda: Z(self.raw_value_int),
				MT_Q: lambda: Q(self.raw_value_rat[0], self.raw_value_rat[1]),
				MT_R: lambda: R(self.raw_value_float),
				MT_C: lambda: (
					self.raw_value_complex[0] + sp.I * self.raw_value_complex[1]
				),
			},
			(2,): {
				MT_Z: lambda: sp.Matrix([Z(i) for i in self.raw_value_int2]),
				MT_Q: lambda: sp.Matrix([Q(q[0], q[1]) for q in self.raw_value_rat2]),
				MT_R: lambda: sp.Matrix([R(r) for r in self.raw_value_float2]),
				MT_C: lambda: sp.Matrix(
					[c[0] + sp.I * c[1] for c in self.raw_value_complex2]
				),
			},
			(3,): {
				MT_Z: lambda: sp.Matrix([Z(i) for i in self.raw_value_int3]),
				MT_Q: lambda: sp.Matrix([Q(q[0], q[1]) for q in self.raw_value_rat3]),
				MT_R: lambda: sp.Matrix([R(r) for r in self.raw_value_float3]),
				MT_C: lambda: sp.Matrix(
					[c[0] + sp.I * c[1] for c in self.raw_value_complex3]
				),
			},
		}[self.shape][self.mathtype]() * (self.unit if self.unit is not None else 1)

	@value.setter
	def value(self, expr: spux.SympyExpr) -> None:
		"""Set the expression defined by the socket.

		Notes:
			Called to set the internal `FlowKind.Value` of this socket.
		"""
		_mathtype, _shape = self._parse_expr_info(expr)
		if self.symbols or self.shape not in [None, (2,), (3,)]:
			self.raw_value_spstr = sp.sstr(expr)

		else:
			MT_Z = spux.MathType.Integer
			MT_Q = spux.MathType.Rational
			MT_R = spux.MathType.Real
			MT_C = spux.MathType.Complex
			if self.shape is None:
				if self.mathtype == MT_Z:
					self.raw_value_int = self._to_raw_value(expr)
				elif self.mathtype == MT_Q:
					self.raw_value_rat = self._to_raw_value(expr)
				elif self.mathtype == MT_R:
					self.raw_value_float = self._to_raw_value(expr)
				elif self.mathtype == MT_C:
					self.raw_value_complex = self._to_raw_value(
						expr, force_complex=True
					)
			elif self.shape == (2,):
				if self.mathtype == MT_Z:
					self.raw_value_int2 = self._to_raw_value(expr)
				elif self.mathtype == MT_Q:
					self.raw_value_rat2 = self._to_raw_value(expr)
				elif self.mathtype == MT_R:
					self.raw_value_float2 = self._to_raw_value(expr)
				elif self.mathtype == MT_C:
					self.raw_value_complex2 = self._to_raw_value(
						expr, force_complex=True
					)
			elif self.shape == (3,):
				if self.mathtype == MT_Z:
					self.raw_value_int3 = self._to_raw_value(expr)
				elif self.mathtype == MT_Q:
					self.raw_value_rat3 = self._to_raw_value(expr)
				elif self.mathtype == MT_R:
					self.raw_value_float3 = self._to_raw_value(expr)
				elif self.mathtype == MT_C:
					self.raw_value_complex3 = self._to_raw_value(
						expr, force_complex=True
					)

	####################
	# - FlowKind: LazyArrayRange
	####################
	@property
	def lazy_array_range(self) -> ct.LazyArrayRangeFlow:
		"""Return the not-yet-computed uniform array defined by the socket.

		Notes:
			Called to compute the internal `FlowKind.LazyArrayRange` of this socket.

		Return:
			The range of lengths, which uses no symbols.
		"""
		if self.symbols:
			return ct.LazyArrayRangeFlow(
				start=self.raw_min_sp,
				stop=self.raw_max_sp,
				steps=self.steps,
				scaling='lin',
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

		return ct.LazyArrayRangeFlow(
			start=min_bound,
			stop=max_bound,
			steps=self.steps,
			scaling='lin',
			unit=self.unit,
		)

	@lazy_array_range.setter
	def lazy_array_range(self, value: ct.LazyArrayRangeFlow) -> None:
		"""Set the not-yet-computed uniform array defined by the socket.

		Notes:
			Called to compute the internal `FlowKind.LazyArrayRange` of this socket.
		"""
		self.steps = value.steps

		if self.symbols:
			self.raw_min_spstr = sp.sstr(value.start)
			self.raw_max_spstr = sp.sstr(value.stop)

		else:
			MT_Z = spux.MathType.Integer
			MT_Q = spux.MathType.Rational
			MT_R = spux.MathType.Real
			MT_C = spux.MathType.Complex

			unit = value.unit if value.unit is not None else 1
			if value.mathtype == MT_Z:
				self.raw_range_int = [
					self._to_raw_value(bound * unit)
					for bound in [value.start, value.stop]
				]
			elif value.mathtype == MT_Q:
				self.raw_range_rat = [
					self._to_raw_value(bound * unit)
					for bound in [value.start, value.stop]
				]
			elif value.mathtype == MT_R:
				self.raw_range_float = [
					self._to_raw_value(bound * unit)
					for bound in [value.start, value.stop]
				]
			elif value.mathtype == MT_C:
				self.raw_range_complex = [
					self._to_raw_value(bound * unit, force_complex=True)
					for bound in [value.start, value.stop]
				]

	####################
	# - FlowKind: LazyValueFunc (w/Params if Constant)
	####################
	@property
	def lazy_value_func(self) -> ct.LazyValueFuncFlow:
		# Lazy Value: Arbitrary Expression
		if self.symbols or self.shape not in [None, (2,), (3,)]:
			return ct.LazyValueFuncFlow(
				func=sp.lambdify(self.sorted_symbols, self.value, 'jax'),
				func_args=[spux.MathType.from_expr(sym) for sym in self.sorted_symbols],
				supports_jax=True,
			)

		# Lazy Value: Constant
		## -> A very simple function, which takes a single argument.
		## -> What will be passed is a unit-scaled/stripped, pytype-converted Expr:Value.
		## -> Until then, the user can utilize this LVF in a function composition chain.
		return ct.LazyValueFuncFlow(
			func=lambda v: v,
			func_args=[
				self.physical_type if self.physical_type is not None else self.mathtype
			],
			supports_jax=True,
		)

	@property
	def params(self) -> ct.ParamsFlow:
		# Params Value: Symbolic
		## -> The Expr socket does not declare actual values for the symbols.
		## -> Those values must come from elsewhere.
		## -> If someone tries to load them anyway, tell them 'NoFlow'.
		if self.symbols or self.shape not in [None, (2,), (3,)]:
			return ct.FlowSignal.NoFlow

		# Params Value: Constant
		## -> Simply pass the Expr:Value as parameter.
		return ct.ParamsFlow(func_args=[self.value])

	####################
	# - FlowKind: Array
	####################
	@property
	def array(self) -> ct.ArrayFlow:
		if not self.symbols:
			return ct.ArrayFlow(
				values=self.lazy_value_func.func_jax(),
				unit=self.unit,
			)

		return ct.FlowSignal.NoFlow

	####################
	# - FlowKind: Info
	####################
	@property
	def info(self) -> ct.ArrayFlow:
		return ct.InfoFlow(
			output_name='_',
			output_shape=self.shape,
			output_mathtype=self.mathtype,
			output_unit=self.unit,
		)

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
	# - UI
	####################
	def draw_label_row(self, row: bpy.types.UILayout, text) -> None:
		if self.active_unit is not None:
			split = row.split(factor=0.6, align=True)

			_row = split.row(align=True)
			_row.label(text=text)

			_col = split.column(align=True)
			_col.prop(self, self.blfields['active_unit'], text='')
		else:
			row.label(text=text)

	def draw_value(self, col: bpy.types.UILayout) -> None:
		if self.symbols:
			col.prop(self, self.blfields['raw_value_spstr'], text='')

		else:
			MT_Z = spux.MathType.Integer
			MT_Q = spux.MathType.Rational
			MT_R = spux.MathType.Real
			MT_C = spux.MathType.Complex
			if self.shape is None:
				if self.mathtype == MT_Z:
					col.prop(self, self.blfields['raw_value_int'], text='')
				elif self.mathtype == MT_Q:
					col.prop(self, self.blfields['raw_value_rat'], text='')
				elif self.mathtype == MT_R:
					col.prop(self, self.blfields['raw_value_float'], text='')
				elif self.mathtype == MT_C:
					col.prop(self, self.blfields['raw_value_complex'], text='')
			elif self.shape == (2,):
				if self.mathtype == MT_Z:
					col.prop(self, self.blfields['raw_value_int2'], text='')
				elif self.mathtype == MT_Q:
					col.prop(self, self.blfields['raw_value_rat2'], text='')
				elif self.mathtype == MT_R:
					col.prop(self, self.blfields['raw_value_float2'], text='')
				elif self.mathtype == MT_C:
					col.prop(self, self.blfields['raw_value_complex2'], text='')
			elif self.shape == (3,):
				if self.mathtype == MT_Z:
					col.prop(self, self.blfields['raw_value_int3'], text='')
				elif self.mathtype == MT_Q:
					col.prop(self, self.blfields['raw_value_rat3'], text='')
				elif self.mathtype == MT_R:
					col.prop(self, self.blfields['raw_value_float3'], text='')
				elif self.mathtype == MT_C:
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

	def draw_lazy_array_range(self, col: bpy.types.UILayout) -> None:
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

		col.prop(self, self.blfields['steps'], text='')

	def draw_input_label_row(self, row: bpy.types.UILayout, text) -> None:
		info = self.compute_data(kind=ct.FlowKind.Info)
		has_dims = not ct.FlowSignal.check(info) and info.dim_names

		if has_dims:
			split = row.split(factor=0.85, align=True)
			_row = split.row(align=False)
		else:
			_row = row

		_row.label(text=text)
		if has_dims:
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

	def draw_info(self, info: ct.InfoFlow, col: bpy.types.UILayout) -> None:
		if self.active_kind == ct.FlowKind.Array and self.show_info_columns:
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
			for dim_name in info.dim_names:
				dim_idx = info.dim_idx[dim_name]
				grid.label(text=dim_name)
				if InfoDisplayCol.Length in self.info_columns:
					grid.label(text=str(len(dim_idx)))
				if InfoDisplayCol.MathType in self.info_columns:
					grid.label(text=spux.MathType.to_str(dim_idx.mathtype))
				if InfoDisplayCol.Unit in self.info_columns:
					grid.label(text=spux.sp_to_str(dim_idx.unit))

			# Outputs
			grid.label(text=info.output_name)
			if InfoDisplayCol.Length in self.info_columns:
				grid.label(text='', icon=ct.Icon.DataSocketOutput)
			if InfoDisplayCol.MathType in self.info_columns:
				grid.label(
					text=(
						spux.MathType.to_str(info.output_mathtype)
						+ (
							'ˣ'.join(
								[
									unicode_superscript(out_axis)
									for out_axis in info.output_shape
								]
							)
							if info.output_shape
							else ''
						)
					)
				)
			if InfoDisplayCol.Unit in self.info_columns:
				grid.label(text=f'{spux.sp_to_str(info.output_unit)}')


####################
# - Socket Configuration
####################
class ExprSocketDef(base.SocketDef):
	socket_type: ct.SocketType = ct.SocketType.Expr
	active_kind: typ.Literal[
		ct.FlowKind.Value, ct.FlowKind.LazyArrayRange, ct.FlowKind.Array
	] = ct.FlowKind.Value

	# Socket Interface
	shape: tuple[int, ...] | None = None
	mathtype: spux.MathType = spux.MathType.Real
	physical_type: spux.PhysicalType | None = None
	symbols: frozenset[spux.Symbol] = frozenset()

	# Socket Units
	default_unit: spux.Unit | None = None

	# FlowKind: Value
	default_value: spux.SympyExpr = sp.S(0)
	abs_min: spux.SympyExpr | None = None
	abs_max: spux.SympyExpr | None = None

	# FlowKind: LazyArrayRange
	default_min: spux.SympyExpr = sp.S(0)
	default_max: spux.SympyExpr = sp.S(1)
	default_steps: int = 2

	# UI
	show_info_columns: bool = False

	####################
	# - Validators - Coersion
	####################
	@pyd.model_validator(mode='after')
	def shape_value_coersion(self) -> str:
		if self.shape is not None and not isinstance(self.default_value, sp.MatrixBase):
			if len(self.shape) == 1:
				self.default_value = self.default_value * sp.Matrix.ones(
					self.shape[0], 1
				)
			if len(self.shape) == 2:
				self.default_value = self.default_value * sp.Matrix.ones(*self.shape)

		return self

	@pyd.model_validator(mode='after')
	def unit_coersion(self) -> str:
		if self.physical_type is not None and self.default_unit is None:
			self.default_unit = self.physical_type.default_unit

		return self

	####################
	# - Validators - Assertion
	####################
	@pyd.model_validator(mode='after')
	def valid_shapes(self) -> str:
		if self.active_kind == ct.FlowKind.LazyArrayRange and self.shape is not None:
			msg = "Can't have a non-None shape when LazyArrayRange is set as the active kind."
			raise ValueError(msg)

		return self

	@pyd.model_validator(mode='after')
	def mathtype_value(self) -> str:
		default_value_mathtype = spux.MathType.from_expr(self.default_value)
		if not self.mathtype.is_compatible(default_value_mathtype):
			msg = f'MathType is {self.mathtype}, but tried to set default value {self.default_value} with mathtype {default_value_mathtype}'
			raise ValueError(msg)

		return self

	@pyd.model_validator(mode='after')
	def symbols_value(self) -> str:
		if (
			self.default_value.free_symbols
			and not self.default_value.free_symbols.issubset(self.symbols)
		):
			msg = f'Tried to set default value {self.default_value} with free symbols {self.default_value.free_symbols}, which is incompatible with socket symbols {self.symbols}'
			raise ValueError(msg)

		return self

	@pyd.model_validator(mode='after')
	def shape_value(self) -> str:
		shape = spux.parse_shape(self.default_value)
		if shape != self.shape and not (
			shape is not None
			and self.shape is not None
			and len(self.shape) == 1
			and 1 in shape
		):
			msg = f'Default value {self.default_value} has shape {shape}, which is incompatible with the expr socket (shape {self.shape})'
			raise ValueError(msg)

		return self

	####################
	# - Initialization
	####################
	def init(self, bl_socket: ExprBLSocket) -> None:
		bl_socket.active_kind = self.active_kind

		# Socket Interface
		bl_socket.shape = self.shape
		bl_socket.mathtype = self.mathtype
		bl_socket.physical_type = self.physical_type
		bl_socket.symbols = self.symbols

		# Socket Units & FlowKind.Value
		if self.physical_type is not None:
			bl_socket.unit = self.default_unit
			bl_socket.value = self.default_value * self.default_unit
		else:
			bl_socket.value = self.default_value

		# FlowKind: LazyArrayRange
		bl_socket.lazy_array_range = ct.LazyArrayRangeFlow(
			start=self.default_min,
			stop=self.default_max,
			steps=self.default_steps,
			scaling='lin',
			unit=self.default_unit,
		)

		# UI
		bl_socket.show_info_columns = self.show_info_columns


####################
# - Blender Registration
####################
BL_REGISTER = [
	ExprBLSocket,
]
