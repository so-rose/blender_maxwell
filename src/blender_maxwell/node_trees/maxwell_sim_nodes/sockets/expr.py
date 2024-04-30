import enum
import typing as typ

import bpy
import sympy as sp

from blender_maxwell.utils import bl_cache, logger
from blender_maxwell.utils import extra_sympy_units as spux

from .. import contracts as ct
from . import base

## TODO: This is a big node, and there's a lot to get right.
## - Dynamically adjust socket color in response to, especially, the unit dimension.
## - Iron out the meaning of display shapes.
## - Generally pay attention to validity checking; it's make or break.
## - For array generation, it may pay to have both a symbolic expression (producing output according to `shape` as usual) denoting how to actually make values, and how many. Enables ex. easy symbolic plots.

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
	socket_type = ct.SocketType.Expr
	bl_label = 'Expr'

	####################
	# - Properties
	####################
	shape: tuple[int, ...] | None = bl_cache.BLField(None)
	mathtype: spux.MathType = bl_cache.BLField(spux.MathType.Real, prop_ui=True)
	physical_type: spux.PhysicalType | None = bl_cache.BLField(None)
	symbols: frozenset[spux.Symbol] = bl_cache.BLField(frozenset())

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
	steps: int = bl_cache.BLField(2, abs_min=2)
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
	def search_units(self, _: bpy.types.Context) -> list[ct.BLEnumElement]:
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
	def unit(self, unit: spux.Unit) -> None:
		"""Set the unit, without touching the `raw_*` UI properties.

		Notes:
			To set a new unit, **and** convert the `raw_*` UI properties to the new unit, use `self.convert_unit()` instead.
		"""
		if unit in self.physical_type.valid_units:
			self.active_unit = sp.sstr(unit)

		msg = f'Tried to set invalid unit {unit} (physical type "{self.physical_type}" only supports "{self.physical_type.valid_units}")'
		raise ValueError(msg)

	def convert_unit(self, unit_to: spux.Unit) -> None:
		if self.active_kind == ct.FlowKind.Value:
			current_value = self.value
			self.unit = unit_to
			self.value = current_value
		elif self.active_kind == ct.FlowKind.LazyArrayRange:
			current_lazy_array_range = self.lazy_array_range
			self.unit = unit_to
			self.lazy_array_range = current_lazy_array_range

	####################
	# - Property Callback
	####################
	def on_socket_prop_changed(self, prop_name: str) -> None:
		if prop_name == 'unit' and self.active_unit is not None:
			self.convert_unit(spux.unit_str_to_unit(self.active_unit))

	####################
	# - Methods
	####################
	def _parse_expr_info(
		self, expr: spux.SympyExpr
	) -> tuple[spux.MathType, tuple[int, ...] | None, spux.UnitDimension]:
		# Parse MathType
		mathtype = spux.MathType.from_expr(expr)
		if self.mathtype != mathtype:
			msg = f'MathType is {self.mathtype}, but tried to set expr {expr} with mathtype {mathtype}'
			raise ValueError(msg)

		# Parse Symbols
		if expr.free_symbols:
			if self.mathtype is not None:
				msg = f'MathType is {self.mathtype}, but tried to set expr {expr} with free symbols {expr.free_symbols}'
				raise ValueError(msg)

			if not expr.free_symbols.issubset(self.symbols):
				msg = f'Tried to set expr {expr} with free symbols {expr.free_symbols}, which is incompatible with socket symbols {self.symbols}'
				raise ValueError(msg)

		# Parse Dimensions
		shape = spux.parse_shape(expr)
		if shape != self.shape:
			msg = f'Expr {expr} has shape {shape}, which is incompatible with the expr socket (shape {self.shape})'
			raise ValueError(msg)

		return mathtype, shape

	def _to_raw_value(self, expr: spux.SympyExpr):
		if self.unit is not None:
			return spux.sympy_to_python(spux.scale_to_unit(expr, self.unit))
		return spux.sympy_to_python(expr)

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
		except ValueError(expr) as ex:
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
		mathtype, shape = self._parse_expr_info(expr)
		if self.symbols or self.shape not in [None, (2,), (3,)]:
			self.raw_value_spstr = sp.sstr(expr)

		else:
			MT_Z = spux.MathType.Integer
			MT_Q = spux.MathType.Rational
			MT_R = spux.MathType.Real
			MT_C = spux.MathType.Complex
			if shape is None:
				if mathtype == MT_Z:
					self.raw_value_int = self._to_raw_value(expr)
				elif mathtype == MT_Q:
					self.raw_value_rat = self._to_raw_value(expr)
				elif mathtype == MT_R:
					self.raw_value_float = self._to_raw_value(expr)
				elif mathtype == MT_C:
					self.raw_value_complex = self._to_raw_value(expr)
			elif shape == (2,):
				if mathtype == MT_Z:
					self.raw_value_int2 = self._to_raw_value(expr)
				elif mathtype == MT_Q:
					self.raw_value_rat2 = self._to_raw_value(expr)
				elif mathtype == MT_R:
					self.raw_value_float2 = self._to_raw_value(expr)
				elif mathtype == MT_C:
					self.raw_value_complex2 = self._to_raw_value(expr)
			elif shape == (3,):
				if mathtype == MT_Z:
					self.raw_value_int3 = self._to_raw_value(expr)
				elif mathtype == MT_Q:
					self.raw_value_rat3 = self._to_raw_value(expr)
				elif mathtype == MT_R:
					self.raw_value_float3 = self._to_raw_value(expr)
				elif mathtype == MT_C:
					self.raw_value_complex3 = self._to_raw_value(expr)

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
		self.unit = value.unit

		if self.symbols:
			self.raw_min_spstr = sp.sstr(value.start)
			self.raw_max_spstr = sp.sstr(value.stop)

		else:
			MT_Z = spux.MathType.Integer
			MT_Q = spux.MathType.Rational
			MT_R = spux.MathType.Real
			MT_C = spux.MathType.Complex

			if value.mathtype == MT_Z:
				self.raw_range_int = [
					self._to_raw_value(bound) for bound in [value.start, value.stop]
				]
			elif value.mathtype == MT_Q:
				self.raw_range_rat = [
					self._to_raw_value(bound) for bound in [value.start, value.stop]
				]
			elif value.mathtype == MT_R:
				self.raw_range_float = [
					self._to_raw_value(bound) for bound in [value.start, value.stop]
				]
			elif value.mathtype == MT_C:
				self.raw_range_complex = [
					self._to_raw_value(bound) for bound in [value.start, value.stop]
				]

	####################
	# - FlowKind: LazyValueFunc (w/Params if Constant)
	####################
	@property
	def lazy_value_func(self) -> ct.LazyValueFuncFlow:
		# Lazy Value: Arbitrary Expression
		if self.symbols or self.shape not in [None, (2,), (3,)]:
			return ct.LazyValueFuncFlow(
				func=sp.lambdify(self.symbols, self.value, 'jax'),
				func_args=[spux.MathType.from_expr(sym) for sym in self.symbols],
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

		msg = "Expr socket can't produce array from expression with free symbols"
		raise ValueError(msg)

	####################
	# - FlowKind: Info
	####################
	@property
	def info(self) -> ct.ArrayFlow:
		return ct.InfoFlow(
			output_name='_',  ## TODO: Something else
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
		## TODO: Prevent all invalid linkage between sockets used as expressions, but don't be too brutal :)
		## - This really is a killer feature. But we want to get it right. So we leave it as todo until we know exactly how to tailor CapabilitiesFlow to these needs.

	####################
	# - UI
	####################
	def draw_label_row(self, row: bpy.types.UILayout, text) -> None:
		if self.active_unit is not None:
			split = row.split(factor=0.6, align=True)

			_row = split.row(align=True)
			_row.label(text=text)

			_col = split.column(align=True)
			_col.prop(self, 'active_unit', text='')

	def draw_value(self, col: bpy.types.UILayout) -> None:
		# Property Interface
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
		if info.dim_names and self.show_info_columns:
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
	active_kind: typ.Literal[ct.FlowKind.Value, ct.FlowKind.LazyArrayRange] = (
		ct.FlowKind.Value
	)

	# Socket Interface
	## TODO: __hash__ like socket method based on these?
	shape: tuple[int, ...] | None = None
	mathtype: spux.MathType = spux.MathType.Real
	physical_type: spux.PhysicalType | None = None
	symbols: frozenset[spux.Symbol] = frozenset()

	# Socket Units
	default_unit: spux.Unit | None = None

	# FlowKind: Value
	default_value: spux.SympyExpr = sp.S(0)

	# FlowKind: LazyArrayRange
	default_min: spux.SympyExpr = sp.S(0)
	default_max: spux.SympyExpr = sp.S(1)
	default_steps: int = 2
	## TODO: Configure lin/log/... scaling (w/enumprop in UI)

	## TODO: Buncha validation :)

	# UI
	show_info_columns: bool = False

	def init(self, bl_socket: ExprBLSocket) -> None:
		bl_socket.active_kind = self.active_kind

		# Socket Interface
		bl_socket.shape = self.shape
		bl_socket.mathtype = self.mathtype
		bl_socket.physical_type = self.physical_type
		bl_socket.symbols = self.symbols

		# Socket Units
		if self.default_unit is not None:
			bl_socket.unit = self.default_unit

		# FlowKind: Value
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
