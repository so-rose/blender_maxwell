"""Declares useful sympy units and functions, to make it easier to work with `sympy` as the basis for a unit-aware system.

Attributes:
	UNIT_BY_SYMBOL: Maps all abbreviated Sympy symbols to their corresponding Sympy unit.
		This is essential for parsing string expressions that use units, since a pure parse of ex. `a*m + m` would not otherwise be able to differentiate between `sp.Symbol(m)` and `spu.meter`.
	SympyType: A simple union of valid `sympy` types, used to check whether arbitrary objects should be handled using `sympy` functions.
		For simple `isinstance` checks, this should be preferred, as it is most performant.
		For general use, `SympyExpr` should be preferred.
	SympyExpr: A `SympyType` that is compatible with `pydantic`, including serialization/deserialization.
		Should be used via the `ConstrSympyExpr`, which also adds expression validation.
"""

import enum
import functools
import typing as typ
from fractions import Fraction

import jax
import jax.numpy as jnp
import pydantic as pyd
import sympy as sp
import sympy.physics.units as spu
import typing_extensions as typx
from pydantic_core import core_schema as pyd_core_schema

from blender_maxwell import contracts as ct

SympyType = (
	sp.Basic
	| sp.Expr
	| sp.MatrixBase
	| sp.MutableDenseMatrix
	| spu.Quantity
	| spu.Dimension
)


####################
# - Math Type
####################
class MathType(enum.StrEnum):
	"""Type identifiers that encompass common sets of mathematical objects."""

	Bool = enum.auto()
	Integer = enum.auto()
	Rational = enum.auto()
	Real = enum.auto()
	Complex = enum.auto()

	def combine(*mathtypes: list[typ.Self]) -> typ.Self:
		if MathType.Complex in mathtypes:
			return MathType.Complex
		elif MathType.Real in mathtypes:
			return MathType.Real
		elif MathType.Rational in mathtypes:
			return MathType.Rational
		elif MathType.Integer in mathtypes:
			return MathType.Integer
		elif MathType.Bool in mathtypes:
			return MathType.Bool

	@staticmethod
	def from_expr(sp_obj: SympyType) -> type:
		## TODO: Support for sp.Matrix
		if isinstance(sp_obj, sp.logic.boolalg.Boolean):
			return MathType.Bool
		if sp_obj.is_integer:
			return MathType.Integer
		if sp_obj.is_rational:
			return MathType.Rational
		if sp_obj.is_real:
			return MathType.Real
		if sp_obj.is_complex:
			return MathType.Complex

		msg = f"Can't determine MathType from sympy object: {sp_obj}"
		raise ValueError(msg)

	@staticmethod
	def from_pytype(dtype) -> type:
		return {
			bool: MathType.Bool,
			int: MathType.Integer,
			float: MathType.Real,
			complex: MathType.Complex,
		}[dtype]

	@property
	def pytype(self) -> type:
		MT = MathType
		return {
			MT.Bool: bool,
			MT.Integer: int,
			MT.Rational: float,
			MT.Real: float,
			MT.Complex: complex,
		}[self]

	@staticmethod
	def to_str(value: typ.Self) -> type:
		return {
			MathType.Bool: 'T|F',
			MathType.Integer: 'ℤ',
			MathType.Rational: 'ℚ',
			MathType.Real: 'ℝ',
			MathType.Complex: 'ℂ',
		}[value]

	@staticmethod
	def to_name(value: typ.Self) -> str:
		return MathType.to_str(value)

	@staticmethod
	def to_icon(value: typ.Self) -> str:
		return ''

	def bl_enum_element(self, i: int) -> ct.BLEnumElement:
		return (
			str(self),
			MathType.to_name(self),
			MathType.to_name(self),
			MathType.to_icon(self),
			i,
		)


class NumberSize1D(enum.StrEnum):
	"""Valid 1D-constrained shape."""

	Scalar = enum.auto()
	Vec2 = enum.auto()
	Vec3 = enum.auto()
	Vec4 = enum.auto()

	@staticmethod
	def to_name(value: typ.Self) -> str:
		NS = NumberSize1D
		return {
			NS.Scalar: 'Scalar',
			NS.Vec2: '2D',
			NS.Vec3: '3D',
			NS.Vec4: '4D',
		}[value]

	@staticmethod
	def to_icon(value: typ.Self) -> str:
		NS = NumberSize1D
		return {
			NS.Scalar: '',
			NS.Vec2: '',
			NS.Vec3: '',
			NS.Vec4: '',
		}[value]

	def bl_enum_element(self, i: int) -> ct.BLEnumElement:
		return (
			str(self),
			NumberSize1D.to_name(self),
			NumberSize1D.to_name(self),
			NumberSize1D.to_icon(self),
			i,
		)

	@staticmethod
	def supports_shape(shape: tuple[int, ...] | None):
		return shape is None or (len(shape) == 1 and shape[0] in [2, 3])

	@staticmethod
	def from_shape(shape: tuple[typ.Literal[2, 3]] | None) -> typ.Self:
		NS = NumberSize1D
		return {
			None: NS.Scalar,
			(2,): NS.Vec2,
			(3,): NS.Vec3,
			(4,): NS.Vec3,
		}[shape]

	@property
	def shape(self):
		NS = NumberSize1D
		return {
			NS.Scalar: None,
			NS.Vec2: (2,),
			NS.Vec3: (3,),
			NS.Vec3: (4,),
		}[self]


####################
# - Unit Dimensions
####################
class DimsMeta(type):
	def __getattr__(cls, attr: str) -> spu.Dimension:
		if (
			attr in spu.definitions.dimension_definitions.__dir__()
			and not attr.startswith('__')
		):
			return getattr(spu.definitions.dimension_definitions, attr)

		raise AttributeError(name=attr, obj=Dims)


class Dims(metaclass=DimsMeta):
	"""Access `sympy.physics.units` dimensions with less hassle.

	Any unit dimension available in `sympy.physics.units.definitions.dimension_definitions` can be accessed as an attribute of `Dims`.

	An `AttributeError` is raised if the unit cannot be found in `sympy`.

	Examples:
		The objects returned are a direct alias to `sympy`, with less hassle:
		```python
		assert Dims.length == (
			sympy.physics.units.definitions.dimension_definitions.length
		)
		```
	"""


####################
# - Units
####################
femtosecond = fs = spu.Quantity('femtosecond', abbrev='fs')
femtosecond.set_global_relative_scale_factor(spu.femto, spu.second)

# Length
femtometer = fm = spu.Quantity('femtometer', abbrev='fm')
femtometer.set_global_relative_scale_factor(spu.femto, spu.meter)

# Lum Flux
lumen = lm = spu.Quantity('lumen', abbrev='lm')
lumen.set_global_relative_scale_factor(1, spu.candela * spu.steradian)

# Force
nanonewton = nN = spu.Quantity('nanonewton', abbrev='nN')  # noqa: N816
nanonewton.set_global_relative_scale_factor(spu.nano, spu.newton)

micronewton = uN = spu.Quantity('micronewton', abbrev='μN')  # noqa: N816
micronewton.set_global_relative_scale_factor(spu.micro, spu.newton)

millinewton = mN = spu.Quantity('micronewton', abbrev='mN')  # noqa: N816
micronewton.set_global_relative_scale_factor(spu.milli, spu.newton)

# Frequency
kilohertz = KHz = spu.Quantity('kilohertz', abbrev='KHz')
kilohertz.set_global_relative_scale_factor(spu.kilo, spu.hertz)

megahertz = MHz = spu.Quantity('megahertz', abbrev='MHz')
kilohertz.set_global_relative_scale_factor(spu.kilo, spu.hertz)

gigahertz = GHz = spu.Quantity('gigahertz', abbrev='GHz')
gigahertz.set_global_relative_scale_factor(spu.giga, spu.hertz)

terahertz = THz = spu.Quantity('terahertz', abbrev='THz')
terahertz.set_global_relative_scale_factor(spu.tera, spu.hertz)

petahertz = PHz = spu.Quantity('petahertz', abbrev='PHz')
petahertz.set_global_relative_scale_factor(spu.peta, spu.hertz)

exahertz = EHz = spu.Quantity('exahertz', abbrev='EHz')
exahertz.set_global_relative_scale_factor(spu.exa, spu.hertz)

# Pressure
millibar = mbar = spu.Quantity('millibar', abbrev='mbar')
millibar.set_global_relative_scale_factor(spu.milli, spu.bar)

hectopascal = hPa = spu.Quantity('hectopascal', abbrev='hPa')  # noqa: N816
hectopascal.set_global_relative_scale_factor(spu.hecto, spu.pascal)

UNIT_BY_SYMBOL: dict[sp.Symbol, spu.Quantity] = {
	unit.name: unit for unit in spu.__dict__.values() if isinstance(unit, spu.Quantity)
} | {unit.name: unit for unit in globals().values() if isinstance(unit, spu.Quantity)}


####################
# - Expr Analysis: Units
####################
## TODO: Caching w/srepr'ed expression.
## TODO: An LFU cache could do better than an LRU.
def uses_units(sp_obj: SympyType) -> bool:
	"""Determines if an expression uses any units.

	Notes:
		The expression graph is traversed depth-first with `sp.postorder_traversal`, to search for `sp.Quantity` elements.
		Depth-first was chosen since `sp.Quantity`s are likelier to be found among individual symbols, rather than complete subexpressions.

		The **worst-case** runtime is when there are no units, in which case the **entire expression graph will be traversed**.

	Parameters:
		expr: The sympy expression that may contain units.

	Returns:
		Whether or not there are units used within the expression.
	"""
	return sp_obj.has(spu.Quantity)
	# return any(
	# isinstance(subexpr, spu.Quantity) for subexpr in sp.postorder_traversal(sp_obj)
	# )


## TODO: Caching w/srepr'ed expression.
## TODO: An LFU cache could do better than an LRU.
def get_units(expr: sp.Expr) -> set[spu.Quantity]:
	"""Finds all units used by the expression, and returns them as a set.

	No information about _the relationship between units_ is exposed.
	For example, compound units like `spu.meter / spu.second` would be mapped to `{spu.meter, spu.second}`.


	Notes:
		The expression graph is traversed depth-first with `sp.postorder_traversal`, to search for `sp.Quantity` elements.

		The performance is comparable to the performance of `sp.postorder_traversal`, since the **entire expression graph will always be traversed**, with the added overhead of one `isinstance` call per expression-graph-node.

	Parameters:
		expr: The sympy expression that may contain units.

	Returns:
		All units (`spu.Quantity`) used within the expression.
	"""
	return {
		subexpr
		for subexpr in sp.postorder_traversal(expr)
		if isinstance(subexpr, spu.Quantity)
	}


def parse_shape(sp_obj: SympyType) -> int | None:
	if isinstance(sp_obj, sp.MatrixBase):
		return sp_obj.shape

	return None


####################
# - Pydantic-Validated SympyExpr
####################
class _SympyExpr:
	"""Low-level `pydantic`, schema describing how to serialize/deserialize fields that have a `SympyType` (like `sp.Expr`), so we can cleanly use `sympy` types in `pyd.BaseModel`.

	Notes:
		You probably want to use `SympyExpr`.

	Examples:
		To be usable as a type annotation on `pyd.BaseModel`, attach this to `SympyType` using `typx.Annotated`:

		```python
		SympyExpr = typx.Annotated[SympyType, _SympyExpr]

		class Spam(pyd.BaseModel):
			line: SympyExpr = sp.Eq(sp.y, 2*sp.Symbol(x, real=True) - 3)
		```
	"""

	@classmethod
	def __get_pydantic_core_schema__(
		cls,
		_source_type: SympyType,
		_handler: pyd.GetCoreSchemaHandler,
	) -> pyd_core_schema.CoreSchema:
		"""Compute a schema that allows `pydantic` to validate a `sympy` type."""

		def validate_from_str(sp_str: str | typ.Any) -> SympyType | typ.Any:
			"""Parse and validate a string expression.

			Parameters:
				sp_str: A stringified `sympy` object, that will be parsed to a sympy type.
					Before use, `isinstance(expr_str, str)` is checked.
					If the object isn't a string, then the validation will be skipped.

			Returns:
				Either a `sympy` object, if the input is parseable, or the same untouched object.

			Raises:
				ValueError: If `sp_str` is a string, but can't be parsed into a `sympy` expression.
			"""
			# Constrain to String
			if not isinstance(sp_str, str):
				return sp_str

			# Parse String -> Sympy
			try:
				expr = sp.sympify(sp_str)
			except ValueError as ex:
				msg = f'String {sp_str} is not a valid sympy expression'
				raise ValueError(msg) from ex

			# Substitute Symbol -> Quantity
			return expr.subs(UNIT_BY_SYMBOL)

		def validate_from_pytype(
			sp_pytype: int | Fraction | float | complex,
		) -> SympyType | typ.Any:
			"""Parse and validate a pure Python type.

			Parameters:
				sp_str: A stringified `sympy` object, that will be parsed to a sympy type.
					Before use, `isinstance(expr_str, str)` is checked.
					If the object isn't a string, then the validation will be skipped.

			Returns:
				Either a `sympy` object, if the input is parseable, or the same untouched object.

			Raises:
				ValueError: If `sp_str` is a string, but can't be parsed into a `sympy` expression.
			"""
			# Constrain to String
			if not isinstance(sp_pytype, int | Fraction | float | complex):
				return sp_pytype

			if isinstance(sp_pytype, int):
				return sp.Integer(sp_pytype)
			if isinstance(sp_pytype, Fraction):
				return sp.Rational(sp_pytype.numerator, sp_pytype.denominator)
			if isinstance(sp_pytype, float):
				return sp.Float(sp_pytype)

			# sp_pytype => Complex
			return sp_pytype.real + sp.I * sp_pytype.imag

		sympy_expr_schema = pyd_core_schema.chain_schema(
			[
				pyd_core_schema.no_info_plain_validator_function(validate_from_str),
				pyd_core_schema.no_info_plain_validator_function(validate_from_pytype),
				pyd_core_schema.is_instance_schema(SympyType),
			]
		)
		return pyd_core_schema.json_or_python_schema(
			json_schema=sympy_expr_schema,
			python_schema=sympy_expr_schema,
			serialization=pyd_core_schema.plain_serializer_function_ser_schema(
				lambda sp_obj: sp.srepr(sp_obj)
			),
		)


SympyExpr = typx.Annotated[
	sp.Basic,  ## Treat all sympy types as sp.Basic
	_SympyExpr,
]
## TODO: The type game between SympyType, SympyExpr, and the various flavors of ConstrSympyExpr(), is starting to be a bit much. Let's consolidate.


def ConstrSympyExpr(  # noqa: N802, PLR0913
	# Features
	allow_variables: bool = True,
	allow_units: bool = True,
	# Structures
	allowed_sets: set[typ.Literal['integer', 'rational', 'real', 'complex']]
	| None = None,
	allowed_structures: set[typ.Literal['scalar', 'matrix']] | None = None,
	# Element Class
	max_symbols: int | None = None,
	allowed_symbols: set[sp.Symbol] | None = None,
	allowed_units: set[spu.Quantity] | None = None,
	# Shape Class
	allowed_matrix_shapes: set[tuple[int, int]] | None = None,
) -> SympyType:
	"""Constructs a `SympyExpr` type, which will validate `sympy` types when used in a `pyd.BaseModel`.

	Relies on the `sympy` assumptions system.
	See <https://docs.sympy.org/latest/guides/assumptions.html#predicates>

	Parameters (TBD):

	Returns:
		A type that represents a constrained `sympy` expression.
	"""

	def validate_expr(expr: SympyType):
		if not (isinstance(expr, SympyType),):
			msg = f"expr '{expr}' is not an allowed Sympy expression ({SympyType})"
			raise ValueError(msg)

		msgs = set()

		# Validate Feature Class
		if (not allow_variables) and (len(expr.free_symbols) > 0):
			msgs.add(
				f'allow_variables={allow_variables} does not match expression {expr}.'
			)
		if (not allow_units) and uses_units(expr):
			msgs.add(f'allow_units={allow_units} does not match expression {expr}.')

		# Validate Structure Class
		if (
			allowed_sets
			and isinstance(expr, sp.Expr)
			and not any(
				{
					'integer': expr.is_integer,
					'rational': expr.is_rational,
					'real': expr.is_real,
					'complex': expr.is_complex,
				}[allowed_set]
				for allowed_set in allowed_sets
			)
		):
			msgs.add(
				f"allowed_sets={allowed_sets} does not match expression {expr} (remember to add assumptions to symbols, ex. `x = sp.Symbol('x', real=True))"
			)
		if allowed_structures and not any(
			{
				'scalar': True,
				'matrix': isinstance(expr, sp.MatrixBase),
			}[allowed_set]
			for allowed_set in allowed_structures
		):
			msgs.add(
				f"allowed_structures={allowed_structures} does not match expression {expr} (remember to add assumptions to symbols, ex. `x = sp.Symbol('x', real=True))"
			)

		# Validate Element Class
		if max_symbols and len(expr.free_symbols) > max_symbols:
			msgs.add(f'max_symbols={max_symbols} does not match expression {expr}')
		if allowed_symbols and expr.free_symbols.issubset(allowed_symbols):
			msgs.add(
				f'allowed_symbols={allowed_symbols} does not match expression {expr}'
			)
		if allowed_units and get_units(expr).issubset(allowed_units):
			msgs.add(f'allowed_units={allowed_units} does not match expression {expr}')

		# Validate Shape Class
		if (
			allowed_matrix_shapes and isinstance(expr, sp.MatrixBase)
		) and expr.shape not in allowed_matrix_shapes:
			msgs.add(
				f'allowed_matrix_shapes={allowed_matrix_shapes} does not match expression {expr} with shape {expr.shape}'
			)

		# Error or Return
		if msgs:
			raise ValueError(str(msgs))
		return expr

	return typx.Annotated[
		sp.Basic,
		_SympyExpr,
		pyd.AfterValidator(validate_expr),
	]


####################
# - Common ConstrSympyExpr
####################
# Expression
ScalarUnitlessRealExpr: typ.TypeAlias = ConstrSympyExpr(
	allow_variables=False,
	allow_units=False,
	allowed_structures={'scalar'},
	allowed_sets={'integer', 'rational', 'real'},
)
ScalarUnitlessComplexExpr: typ.TypeAlias = ConstrSympyExpr(
	allow_variables=False,
	allow_units=False,
	allowed_structures={'scalar'},
	allowed_sets={'integer', 'rational', 'real', 'complex'},
)

# Symbol
IntSymbol: typ.TypeAlias = ConstrSympyExpr(
	allow_variables=True,
	allow_units=False,
	allowed_sets={'integer'},
	max_symbols=1,
)
RationalSymbol: typ.TypeAlias = ConstrSympyExpr(
	allow_variables=True,
	allow_units=False,
	allowed_sets={'integer', 'rational'},
	max_symbols=1,
)
RealSymbol: typ.TypeAlias = ConstrSympyExpr(
	allow_variables=True,
	allow_units=False,
	allowed_sets={'integer', 'rational', 'real'},
	max_symbols=1,
)
ComplexSymbol: typ.TypeAlias = ConstrSympyExpr(
	allow_variables=True,
	allow_units=False,
	allowed_sets={'integer', 'rational', 'real', 'complex'},
	max_symbols=1,
)
Symbol: typ.TypeAlias = IntSymbol | RealSymbol | ComplexSymbol

# Unit
UnitDimension: typ.TypeAlias = SympyExpr  ## Actually spu.Dimension

## Technically a "unit expression", which includes compound types.
## Support for this is the reason to prefer over raw spu.Quantity.
Unit: typ.TypeAlias = ConstrSympyExpr(
	allow_variables=False,
	allow_units=True,
	allowed_structures={'scalar'},
)

# Number
IntNumber: typ.TypeAlias = ConstrSympyExpr(
	allow_variables=False,
	allow_units=False,
	allowed_sets={'integer'},
	allowed_structures={'scalar'},
)
RealNumber: typ.TypeAlias = ConstrSympyExpr(
	allow_variables=False,
	allow_units=False,
	allowed_sets={'integer', 'rational', 'real'},
	allowed_structures={'scalar'},
)
ComplexNumber: typ.TypeAlias = ConstrSympyExpr(
	allow_variables=False,
	allow_units=False,
	allowed_sets={'integer', 'rational', 'real', 'complex'},
	allowed_structures={'scalar'},
)
Number: typ.TypeAlias = IntNumber | RealNumber | ComplexNumber

# Number
PhysicalRealNumber: typ.TypeAlias = ConstrSympyExpr(
	allow_variables=False,
	allow_units=True,
	allowed_sets={'integer', 'rational', 'real'},
	allowed_structures={'scalar'},
)
PhysicalComplexNumber: typ.TypeAlias = ConstrSympyExpr(
	allow_variables=False,
	allow_units=True,
	allowed_sets={'integer', 'rational', 'real', 'complex'},
	allowed_structures={'scalar'},
)
PhysicalNumber: typ.TypeAlias = PhysicalRealNumber | PhysicalComplexNumber

# Vector
Real3DVector: typ.TypeAlias = ConstrSympyExpr(
	allow_variables=False,
	allow_units=False,
	allowed_sets={'integer', 'rational', 'real'},
	allowed_structures={'matrix'},
	allowed_matrix_shapes={(3, 1)},
)


####################
# - Sympy Utilities: Printing
####################
_SYMPY_EXPR_PRINTER_STR = sp.printing.str.StrPrinter(
	settings={
		'abbrev': True,
	}
)


def sp_to_str(sp_obj: SympyExpr) -> str:
	"""Converts a sympy object to an output-oriented string (w/abbreviated units), using a dedicated StrPrinter.

	This should be used whenever a **string for UI use** is needed from a `sympy` object.

	Notes:
		This should **NOT** be used in cases where the string will be `sp.sympify()`ed back into a sympy expression.
		For such cases, rely on `sp.srepr()`, which uses an _explicit_ representation.

	Parameters:
		sp_obj: The `sympy` object to convert to a string.

	Returns:
		A string representing the expression for human use.
			_The string is not re-encodable to the expression._
	"""
	return _SYMPY_EXPR_PRINTER_STR.doprint(sp_obj)


def pretty_symbol(sym: sp.Symbol) -> str:
	return f'{sym.name} ∈ ' + (
		'ℂ'
		if sym.is_complex
		else ('ℝ' if sym.is_real else ('ℤ' if sym.is_integer else '?'))
	)


####################
# - Unit Utilities
####################
def scale_to_unit(sp_obj: SympyType, unit: spu.Quantity) -> Number:
	"""Convert an expression that uses units to a different unit, then strip all units, leaving only a unitless `sympy` value.

	This is used whenever the unitless part of an expression is needed, but guaranteed expressed in a particular unit, aka. **unit system normalization**.

	Notes:
		The unitless output is still an `sp.Expr`, which may contain ex. symbols.

		If you know that the output **should** work as a corresponding Python type (ex. `sp.Integer` vs. `int`), but it doesn't, you can use `sympy_to_python()` to produce a pure-Python type.
		In this way, with a little care, broad compatiblity can be bridged between the `sympy.physics.units` unit system and the wider Python ecosystem.

	Parameters:
		expr: The unit-containing expression to convert.
		unit_to: The unit that is converted to.

	Returns:
		The unitless part of `expr`, after scaling the entire expression to `unit`.

	Raises:
		ValueError: If the result of unit-conversion and -stripping still has units, as determined by `uses_units()`.
	"""
	## TODO: An LFU cache could do better than an LRU.
	unitless_expr = spu.convert_to(sp_obj, unit) / unit
	if not uses_units(unitless_expr):
		return unitless_expr

	msg = f'Sympy object "{sp_obj}" was scaled to the unit "{unit}" with the expectation that the result would be unitless, but the result "{unitless_expr}" has units "{get_units(unitless_expr)}"'
	raise ValueError(msg)


def scaling_factor(unit_from: spu.Quantity, unit_to: spu.Quantity) -> Number:
	"""Compute the numerical scaling factor imposed on the unitless part of the expression when converting from one unit to another.

	Parameters:
		unit_from: The unit that is converted from.
		unit_to: The unit that is converted to.

	Returns:
		The numerical scaling factor between the two units.

	Raises:
		ValueError: If the two units don't share a common dimension.
	"""
	if unit_from.dimension == unit_to.dimension:
		return scale_to_unit(unit_from, unit_to)

	msg = f"Dimension of unit_from={unit_from} ({unit_from.dimension}) doesn't match the dimension of unit_to={unit_to} ({unit_to.dimension}); therefore, there is no scaling factor between them"
	raise ValueError(msg)


_UNIT_STR_MAP = {sym.name: unit for sym, unit in UNIT_BY_SYMBOL.items()}


@functools.cache
def unit_str_to_unit(unit_str: str) -> Unit | None:
	if unit_str in _UNIT_STR_MAP:
		return _UNIT_STR_MAP[unit_str]

	msg = 'No valid unit for unit string {unit_str}'
	raise ValueError(msg)


####################
# - "Physical" Type
####################
class PhysicalType(enum.StrEnum):
	"""Type identifiers for expressions with both `MathType` and a unit, aka a "physical" type."""

	# Global
	Time = enum.auto()
	Angle = enum.auto()
	SolidAngle = enum.auto()
	## TODO: Some kind of 3D-specific orientation ex. a quaternion
	Freq = enum.auto()
	AngFreq = enum.auto()  ## rad*hertz
	# Cartesian
	Length = enum.auto()
	Area = enum.auto()
	Volume = enum.auto()
	# Mechanical
	Vel = enum.auto()
	Accel = enum.auto()
	Mass = enum.auto()
	Force = enum.auto()
	Pressure = enum.auto()
	# Energy
	Work = enum.auto()  ## joule
	Power = enum.auto()  ## watt
	PowerFlux = enum.auto()  ## watt
	Temp = enum.auto()
	# Electrodynamics
	Current = enum.auto()  ## ampere
	CurrentDensity = enum.auto()
	Charge = enum.auto()  ## coulomb
	Voltage = enum.auto()
	Capacitance = enum.auto()  ## farad
	Impedance = enum.auto()  ## ohm
	Conductance = enum.auto()  ## siemens
	Conductivity = enum.auto()  ## siemens / length
	MFlux = enum.auto()  ## weber
	MFluxDensity = enum.auto()  ## tesla
	Inductance = enum.auto()  ## henry
	EField = enum.auto()
	HField = enum.auto()
	# Luminal
	LumIntensity = enum.auto()
	LumFlux = enum.auto()
	Luminance = enum.auto()
	Illuminance = enum.auto()
	# Optics
	OrdinaryWaveVector = enum.auto()
	AngularWaveVector = enum.auto()
	PoyntingVector = enum.auto()

	@property
	def unit_dim(self):
		PT = PhysicalType
		return {
			# Global
			PT.Time: Dims.time,
			PT.Angle: Dims.angle,
			PT.SolidAngle: Dims.steradian,  ## MISSING
			PT.Freq: Dims.frequency,
			PT.AngFreq: Dims.angle * Dims.frequency,
			# Cartesian
			PT.Length: Dims.length,
			PT.Area: Dims.length**2,
			PT.Volume: Dims.length**3,
			# Mechanical
			PT.Vel: Dims.length / Dims.time,
			PT.Accel: Dims.length / Dims.time**2,
			PT.Mass: Dims.mass,
			PT.Force: Dims.force,
			PT.Pressure: Dims.pressure,
			# Energy
			PT.Work: Dims.energy,
			PT.Power: Dims.power,
			PT.PowerFlux: Dims.power / Dims.length**2,
			PT.Temp: Dims.temperature,
			# Electrodynamics
			PT.Current: Dims.current,
			PT.CurrentDensity: Dims.current / Dims.length**2,
			PT.Charge: Dims.charge,
			PT.Voltage: Dims.voltage,
			PT.Capacitance: Dims.capacitance,
			PT.Impedance: Dims.impedance,
			PT.Conductance: Dims.conductance,
			PT.Conductivity: Dims.conductance / Dims.length,
			PT.MFlux: Dims.magnetic_flux,
			PT.MFluxDensity: Dims.magnetic_density,
			PT.Inductance: Dims.inductance,
			PT.EField: Dims.voltage / Dims.length,
			PT.HField: Dims.current / Dims.length,
			# Luminal
			PT.LumIntensity: Dims.luminous_intensity,
			PT.LumFlux: Dims.luminous_intensity * Dims.steradian,
			PT.Illuminance: Dims.luminous_intensity / Dims.length**2,
			# Optics
			PT.OrdinaryWaveVector: Dims.frequency,
			PT.AngularWaveVector: Dims.angle * Dims.frequency,
			PT.PoyntingVector: Dims.power / Dims.length**2,
		}

	@property
	def default_unit(self) -> list[Unit]:
		PT = PhysicalType
		return {
			# Global
			PT.Time: spu.picosecond,
			PT.Angle: spu.radian,
			PT.SolidAngle: spu.steradian,
			PT.Freq: terahertz,
			PT.AngFreq: spu.radian * terahertz,
			# Cartesian
			PT.Length: spu.micrometer,
			PT.Area: spu.um**2,
			PT.Volume: spu.um**3,
			# Mechanical
			PT.Vel: spu.um / spu.second,
			PT.Accel: spu.um / spu.second,
			PT.Mass: spu.microgram,
			PT.Force: micronewton,
			PT.Pressure: millibar,
			# Energy
			PT.Work: spu.joule,
			PT.Power: spu.watt,
			PT.PowerFlux: spu.watt / spu.meter**2,
			PT.Temp: spu.kelvin,
			# Electrodynamics
			PT.Current: spu.ampere,
			PT.CurrentDensity: spu.ampere / spu.meter**2,
			PT.Charge: spu.coulomb,
			PT.Voltage: spu.volt,
			PT.Capacitance: spu.farad,
			PT.Impedance: spu.ohm,
			PT.Conductance: spu.siemens,
			PT.Conductivity: spu.siemens / spu.micrometer,
			PT.MFlux: spu.weber,
			PT.MFluxDensity: spu.tesla,
			PT.Inductance: spu.henry,
			PT.EField: spu.volt / spu.micrometer,
			PT.HField: spu.ampere / spu.micrometer,
			# Luminal
			PT.LumIntensity: spu.candela,
			PT.LumFlux: spu.candela * spu.steradian,
			PT.Illuminance: spu.candela / spu.meter**2,
			# Optics
			PT.OrdinaryWaveVector: terahertz,
			PT.AngularWaveVector: spu.radian * terahertz,
		}[self]

	@property
	def valid_units(self) -> list[Unit]:
		PT = PhysicalType
		return {
			# Global
			PT.Time: [
				femtosecond,
				spu.picosecond,
				spu.nanosecond,
				spu.microsecond,
				spu.millisecond,
				spu.second,
				spu.minute,
				spu.hour,
				spu.day,
			],
			PT.Angle: [
				spu.radian,
				spu.degree,
			],
			PT.SolidAngle: [
				spu.steradian,
			],
			PT.Freq: (
				_valid_freqs := [
					spu.hertz,
					kilohertz,
					megahertz,
					gigahertz,
					terahertz,
					petahertz,
					exahertz,
				]
			),
			PT.AngFreq: [spu.radian * _unit for _unit in _valid_freqs],
			# Cartesian
			PT.Length: (
				_valid_lens := [
					spu.picometer,
					spu.angstrom,
					spu.nanometer,
					spu.micrometer,
					spu.millimeter,
					spu.centimeter,
					spu.meter,
					spu.inch,
					spu.foot,
					spu.yard,
					spu.mile,
				]
			),
			PT.Area: [_unit**2 for _unit in _valid_lens],
			PT.Volume: [_unit**3 for _unit in _valid_lens],
			# Mechanical
			PT.Vel: [_unit / spu.second for _unit in _valid_lens],
			PT.Accel: [_unit / spu.second**2 for _unit in _valid_lens],
			PT.Mass: [
				spu.electron_rest_mass,
				spu.dalton,
				spu.microgram,
				spu.milligram,
				spu.gram,
				spu.kilogram,
				spu.metric_ton,
			],
			PT.Force: [
				spu.kg * spu.meter / spu.second**2,
				nanonewton,
				micronewton,
				millinewton,
				spu.newton,
			],
			PT.Pressure: [
				millibar,
				spu.bar,
				spu.pascal,
				hectopascal,
				spu.atmosphere,
				spu.psi,
				spu.mmHg,
				spu.torr,
			],
			# Energy
			PT.Work: [
				spu.electronvolt,
				spu.joule,
			],
			PT.Power: [
				spu.watt,
			],
			PT.PowerFlux: [
				spu.watt / spu.meter**2,
			],
			PT.Temp: [
				spu.kelvin,
			],
			# Electrodynamics
			PT.Current: [
				spu.ampere,
			],
			PT.CurrentDensity: [
				spu.ampere / spu.meter**2,
			],
			PT.Charge: [
				spu.coulomb,
			],
			PT.Voltage: [
				spu.volt,
			],
			PT.Capacitance: [
				spu.farad,
			],
			PT.Impedance: [
				spu.ohm,
			],
			PT.Conductance: [
				spu.siemens,
			],
			PT.Conductivity: [
				spu.siemens / spu.micrometer,
				spu.siemens / spu.meter,
			],
			PT.MFlux: [
				spu.weber,
			],
			PT.MFluxDensity: [
				spu.tesla,
			],
			PT.Inductance: [
				spu.henry,
			],
			PT.EField: [
				spu.volt / spu.micrometer,
				spu.volt / spu.meter,
			],
			PT.HField: [
				spu.ampere / spu.micrometer,
				spu.ampere / spu.meter,
			],
			# Luminal
			PT.LumIntensity: [
				spu.candela,
			],
			PT.LumFlux: [
				spu.candela * spu.steradian,
			],
			PT.Illuminance: [
				spu.candela / spu.meter**2,
			],
			# Optics
			PT.OrdinaryWaveVector: _valid_freqs,
			PT.AngularWaveVector: [spu.radian * _unit for _unit in _valid_freqs],
		}[self]

	@staticmethod
	def from_unit(unit: Unit) -> list[Unit]:
		for physical_type in list[PhysicalType]:
			if unit in physical_type.valid_units:
				return physical_type

		msg = f'No PhysicalType found for unit {unit}'
		raise ValueError(msg)

	@property
	def valid_shapes(self):
		PT = PhysicalType
		overrides = {
			# Cartesian
			PT.Length: [None, (2,), (3,)],
			# Mechanical
			PT.Vel: [None, (2,), (3,)],
			PT.Accel: [None, (2,), (3,)],
			PT.Force: [None, (2,), (3,)],
			# Energy
			PT.Work: [None, (2,), (3,)],
			PT.PowerFlux: [None, (2,), (3,)],
			# Electrodynamics
			PT.CurrentDensity: [None, (2,), (3,)],
			PT.MFluxDensity: [None, (2,), (3,)],
			PT.EField: [None, (2,), (3,)],
			PT.HField: [None, (2,), (3,)],
			# Luminal
			PT.LumFlux: [None, (2,), (3,)],
			# Optics
			PT.OrdinaryWaveVector: [None, (2,), (3,)],
			PT.AngularWaveVector: [None, (2,), (3,)],
			PT.PoyntingVector: [None, (2,), (3,)],
		}

		return overrides.get(self, [None])

	@property
	def valid_mathtypes(self) -> list[MathType]:
		"""Returns a list of valid mathematical types, especially whether it can be real- or complex-valued.

		Generally, all unit quantities are real, in the algebraic mathematical sense.
		However, in electrodynamics especially, it becomes enormously useful to bake in a _rotational component_ as an imaginary value, be it simply to model phase or oscillation-oriented dampening.
		This imaginary part has physical meaning, which can be expressed using the same mathematical formalism associated with unit systems.
		In general, the value is a phasor.

		While it is difficult to arrive at a well-defined way of saying, "this is when a quantity is complex", an attempt has been made to form a sensible baseline based on when phasor math may apply.

		Notes:
			- **Freq**/**AngFreq**: The imaginary part represents growth/dampening of the oscillation.
			- **Current**/**Voltage**: The imaginary part represents the phase.
				This also holds for any downstream units.
			- **Charge**: Generally, it is real.
				However, an imaginary phase term seems to have research applications when dealing with high-order harmonics in high-energy pulsed lasers: <https://iopscience.iop.org/article/10.1088/1361-6455/aac787>
			- **Conductance**: The imaginary part represents the extinction, in the Drude-model sense.
			- **Poynting**: The imaginary part represents the oscillation in the power flux over time.

		"""
		MT = MathType
		PT = PhysicalType
		overrides = {
			# Cartesian
			PT.Freq: [MT.Real, MT.Complex],  ## Im -> Growth/Damping
			PT.AngFreq: [MT.Real, MT.Complex],  ## Im -> Growth/Damping
			# Mechanical
			# Energy
			# Electrodynamics
			PT.Current: [MT.Real, MT.Complex],  ## Im -> Phase
			PT.CurrentDensity: [MT.Real, MT.Complex],  ## Im -> Phase
			PT.Charge: [MT.Real, MT.Complex],  ## Im -> Phase
			PT.Voltage: [MT.Real, MT.Complex],  ## Im -> Phase
			PT.Capacitance: [MT.Real, MT.Complex],  ## Im -> Phase
			PT.Impedance: [MT.Real, MT.Complex],  ## Im -> Reactance
			PT.Inductance: [MT.Real, MT.Complex],  ## Im -> Extinction
			PT.Conductance: [MT.Real, MT.Complex],  ## Im -> Extinction
			PT.Conductivity: [MT.Real, MT.Complex],  ## Im -> Extinction
			PT.MFlux: [MT.Real, MT.Complex],  ## Im -> Phase
			PT.MFluxDensity: [MT.Real, MT.Complex],  ## Im -> Phase
			PT.EField: [MT.Real, MT.Complex],  ## Im -> Phase
			PT.HField: [MT.Real, MT.Complex],  ## Im -> Phase
			# Luminal
			# Optics
			PT.OrdinaryWaveVector: [MT.Real, MT.Complex],  ## Im -> Phase
			PT.AngularWaveVector: [MT.Real, MT.Complex],  ## Im -> Phase
			PT.PoyntingVector: [MT.Real, MT.Complex],  ## Im -> Reactive Power
		}

		return overrides.get(self, [MT.Real])

	@staticmethod
	def to_name(value: typ.Self) -> str:
		return sp_to_str(value.unit_dim)

	@staticmethod
	def to_icon(value: typ.Self) -> str:
		return ''

	def bl_enum_element(self, i: int) -> ct.BLEnumElement:
		PT = PhysicalType
		return (
			str(self),
			PT.to_name(self),
			PT.to_name(self),
			PT.to_icon(self),
			i,
		)


####################
# - Standard Unit Systems
####################
UnitSystem: typ.TypeAlias = dict[PhysicalType, Unit]

_PT = PhysicalType
UNITS_SI: UnitSystem = {
	# Global
	_PT.Time: spu.second,
	_PT.Angle: spu.radian,
	_PT.SolidAngle: spu.steradian,
	_PT.Freq: spu.hertz,
	_PT.AngFreq: spu.radian * spu.hertz,
	# Cartesian
	_PT.Length: spu.meter,
	_PT.Area: spu.meter**2,
	_PT.Volume: spu.meter**3,
	# Mechanical
	_PT.Vel: spu.meter / spu.second,
	_PT.Accel: spu.meter / spu.second**2,
	_PT.Mass: spu.kilogram,
	_PT.Force: spu.newton,
	# Energy
	_PT.Work: spu.joule,
	_PT.Power: spu.watt,
	_PT.PowerFlux: spu.watt / spu.meter**2,
	_PT.Temp: spu.kelvin,
	# Electrodynamics
	_PT.Current: spu.ampere,
	_PT.CurrentDensity: spu.ampere / spu.meter**2,
	_PT.Capacitance: spu.farad,
	_PT.Impedance: spu.ohm,
	_PT.Conductance: spu.siemens,
	_PT.Conductivity: spu.siemens / spu.meter,
	_PT.MFlux: spu.weber,
	_PT.MFluxDensity: spu.tesla,
	_PT.Inductance: spu.henry,
	_PT.EField: spu.volt / spu.meter,
	_PT.HField: spu.ampere / spu.meter,
	# Luminal
	_PT.LumIntensity: spu.candela,
	_PT.LumFlux: lumen,
	_PT.Illuminance: spu.lux,
	# Optics
	_PT.OrdinaryWaveVector: spu.hertz,
	_PT.AngularWaveVector: spu.radian * spu.hertz,
	_PT.PoyntingVector: spu.watt / spu.meter**2,
}


####################
# - Sympy Utilities: Cast to Python
####################
def sympy_to_python(
	scalar: sp.Basic, use_jax_array: bool = False
) -> int | float | complex | tuple | jax.Array:
	"""Convert a scalar sympy expression to the directly corresponding Python type.

	Arguments:
		scalar: A sympy expression that has no symbols, but is expressed as a Sympy type.
			For expressions that are equivalent to a scalar (ex. "(2a + a)/a"), you must simplify the expression with ex. `sp.simplify()` before passing to this parameter.

	Returns:
		A pure Python type that directly corresponds to the input scalar expression.
	"""
	if isinstance(scalar, sp.MatrixBase):
		# Detect Single Column Vector
		## --> Flatten to Single Row Vector
		if len(scalar.shape) == 2 and scalar.shape[1] == 1:
			_scalar = scalar.T
		else:
			_scalar = scalar

		# Convert to Tuple of Tuples
		matrix = tuple(
			[tuple([sympy_to_python(el) for el in row]) for row in _scalar.tolist()]
		)

		# Detect Single Row Vector
		## --> This could be because the scalar had it.
		## --> This could also be because we flattened a column vector.
		## Either way, we should strip the pointless dimensions.
		if len(matrix) == 1:
			return matrix[0] if not use_jax_array else jnp.array(matrix[0])

		return matrix if not use_jax_array else jnp.array(matrix)
	if scalar.is_integer:
		return int(scalar)
	if scalar.is_rational or scalar.is_real:
		return float(scalar)
	if scalar.is_complex:
		return complex(scalar)

	msg = f'Cannot convert sympy scalar expression "{scalar}" to a Python type. Check the assumptions on the expr (current expr assumptions: "{scalar._assumptions}")'  # noqa: SLF001
	raise ValueError(msg)


####################
# - Convert to Unit System
####################
def _flat_unit_system_units(unit_system: UnitSystem) -> SympyExpr:
	return list(unit_system.values())


def convert_to_unit_system(sp_obj: SympyExpr, unit_system: UnitSystem) -> SympyExpr:
	"""Convert an expression to the units of a given unit system, with appropriate scaling."""
	return spu.convert_to(sp_obj, _flat_unit_system_units(unit_system))


def scale_to_unit_system(
	sp_obj: SympyExpr, unit_system: UnitSystem, use_jax_array: bool = False
) -> int | float | complex | tuple | jax.Array:
	"""Convert an expression to the units of a given unit system, then strip all units of the unit system.

	Unit stripping is "dumb": Substitute any `sympy` object in `unit_system.values()` with `1`.
	Afterwards, it is converted to an appropriate Python type.

	Notes:
		For stability, and performance, reasons, this should only be used at the very last stage.

		Regarding performance: **This is not a fast function**.

	Parameters:
		sp_obj: An arbitrary sympy object, presumably with units.
		unit_system: A unit system mapping `PhysicalType` to particular choices of (compound) units.
			Note that, in this context, only `unit_system.values()` is used.

	Returns:
		An appropriate pure Python type, after scaling to the unit system and stripping all units away.

		If the returned type is array-like, and `use_jax_array` is specified, then (and **only** then) will a `jax.Array` be returned instead of a nested `tuple`.
	"""
	return sympy_to_python(
		convert_to_unit_system(sp_obj, unit_system).subs(
			{unit: 1 for unit in unit_system.values()}
		),
		use_jax_array=use_jax_array,
	)
