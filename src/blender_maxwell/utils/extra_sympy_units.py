"""Declares useful sympy units and functions, to make it easier to work with `sympy` as the basis for a unit-aware system.

Attributes:
	ALL_UNIT_SYMBOLS: Maps all abbreviated Sympy symbols to their corresponding Sympy unit.
		This is essential for parsing string expressions that use units, since a pure parse of ex. `a*m + m` would not otherwise be able to differentiate between `sp.Symbol(m)` and `spu.meter`.
	SympyType: A simple union of valid `sympy` types, used to check whether arbitrary objects should be handled using `sympy` functions.
		For simple `isinstance` checks, this should be preferred, as it is most performant.
		For general use, `SympyExpr` should be preferred.
	SympyExpr: A `SympyType` that is compatible with `pydantic`, including serialization/deserialization.
		Should be used via the `ConstrSympyExpr`, which also adds expression validation.
"""

import enum
import itertools
import typing as typ

import jax.numpy as jnp
import pydantic as pyd
import sympy as sp
import sympy.physics.units as spu
import typing_extensions as typx
from pydantic_core import core_schema as pyd_core_schema

SympyType = sp.Basic | sp.Expr | sp.MatrixBase | sp.MutableDenseMatrix | spu.Quantity


class MathType(enum.StrEnum):
	Bool = enum.auto()
	Integer = enum.auto()
	Rational = enum.auto()
	Real = enum.auto()
	Complex = enum.auto()

	@staticmethod
	def from_expr(sp_obj: SympyType) -> type:
		if isinstance(sp_obj, sp.logic.boolalg.Boolean):
			return MathType.Bool
		if sp_obj.is_integer:
			return MathType.Integer
		if sp_obj.is_rational or sp_obj.is_real:
			return MathType.Real
		if sp_obj.is_complex:
			return MathType.Complex

		msg = "Can't determine MathType from sympy object: {sp_obj}"
		raise ValueError(msg)

	@staticmethod
	def from_pytype(dtype) -> type:
		return {
			bool: MathType.Bool,
			int: MathType.Integer,
			float: MathType.Real,
			complex: MathType.Complex,
			#jnp.int32: MathType.Integer,
			#jnp.int64: MathType.Integer,
			#jnp.float32: MathType.Real,
			#jnp.float64: MathType.Real,
			#jnp.complex64: MathType.Complex,
			#jnp.complex128: MathType.Complex,
			#jnp.bool_: MathType.Bool,
		}[dtype]

	@staticmethod
	def to_dtype(value: typ.Self) -> type:
		return {
			MathType.Bool: bool,
			MathType.Integer: int,
			MathType.Rational: float,
			MathType.Real: float,
			MathType.Complex: complex,
		}[value]

	@staticmethod
	def to_str(value: typ.Self) -> type:
		return {
			MathType.Bool: 'T|F',
			MathType.Integer: 'ℤ',
			MathType.Rational: 'ℚ',
			MathType.Real: 'ℝ',
			MathType.Complex: 'ℂ',
		}[value]


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


####################
# - Sympy Printer
####################
_SYMPY_EXPR_PRINTER_STR = sp.printing.str.StrPrinter(
	settings={
		'abbrev': True,
	}
)


def sp_to_str(sp_obj: SympyType) -> str:
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


####################
# - Expr Analysis: Units
####################
## TODO: Caching w/srepr'ed expression.
## TODO: An LFU cache could do better than an LRU.
def uses_units(expr: sp.Expr) -> bool:
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
	return any(
		isinstance(subexpr, spu.Quantity) for subexpr in sp.postorder_traversal(expr)
	)


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


####################
# - Sympy Expression Typing
####################
ALL_UNIT_SYMBOLS: dict[sp.Symbol, spu.Quantity] = {
	unit.name: unit for unit in spu.__dict__.values() if isinstance(unit, spu.Quantity)
} | {unit.name: unit for unit in globals().values() if isinstance(unit, spu.Quantity)}


####################
# - Units <-> Scalars
####################
def scale_to_unit(expr: sp.Expr, unit: spu.Quantity) -> sp.Expr:
	"""Convert an expression that uses units to a different unit, then strip all units.

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
	unitless_expr = spu.convert_to(expr, unit) / unit
	if not uses_units(unitless_expr):
		return unitless_expr

	msg = f'Expression "{expr}" was scaled to the unit "{unit}" with the expectation that the result would be unitless, but the result "{unitless_expr}" has units "{get_units(unitless_expr)}"'
	raise ValueError(msg)


def scaling_factor(unit_from: spu.Quantity, unit_to: spu.Quantity) -> sp.Number:
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


####################
# - Sympy -> Python
####################
## TODO: Integrate SympyExpr for constraining to the output types.
def sympy_to_python_type(sym: sp.Symbol) -> type:
	"""Retrieve the Python type that is implied by a scalar `sympy` symbol.

	Arguments:
		sym: A scalar sympy symbol.

	Returns:
		A pure Python type.
	"""
	if sym.is_integer:
		return int
	if sym.is_rational or sym.is_real:
		return float
	if sym.is_complex:
		return complex

	msg = f'Cannot find Python type for sympy symbol "{sym}". Check the assumptions on the expr (current expr assumptions: "{sym._assumptions}")'  # noqa: SLF001
	raise ValueError(msg)


def sympy_to_python(scalar: sp.Basic) -> int | float | complex | tuple | list:
	"""Convert a scalar sympy expression to the directly corresponding Python type.

	Arguments:
		scalar: A sympy expression that has no symbols, but is expressed as a Sympy type.
			For expressions that are equivalent to a scalar (ex. "(2a + a)/a"), you must simplify the expression with ex. `sp.simplify()` before passing to this parameter.

	Returns:
		A pure Python type that directly corresponds to the input scalar expression.
	"""
	if isinstance(scalar, sp.MatrixBase):
		list_2d = [[sympy_to_python(el) for el in row] for row in scalar.tolist()]

		# Detect Row / Column Vector
		## When it's "actually" a 1D structure, flatten and return as tuple.
		if 1 in scalar.shape:
			return tuple(itertools.chain.from_iterable(list_2d))

		return list_2d
	if scalar.is_integer:
		return int(scalar)
	if scalar.is_rational or scalar.is_real:
		return float(scalar)
	if scalar.is_complex:
		return complex(scalar)

	msg = f'Cannot convert sympy scalar expression "{scalar}" to a Python type. Check the assumptions on the expr (current expr assumptions: "{scalar._assumptions}")'  # noqa: SLF001
	raise ValueError(msg)


def pretty_symbol(sym: sp.Symbol) -> str:
	return f'{sym.name} ∈ ' + (
		'ℂ'
		if sym.is_complex
		else ('ℝ' if sym.is_real else ('ℤ' if sym.is_integer else '?'))
	)


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
			return expr.subs(ALL_UNIT_SYMBOLS)

		# def validate_from_expr(sp_obj: SympyType) -> SympyType:
		# """Validate that a `sympy` object is a `SympyType`.

		# In the static sense, this is a dummy function.

		# Parameters:
		# sp_obj: A `sympy` object.

		# Returns:
		# The `sympy` object.

		# Raises:
		# ValueError: If `sp_obj` is not a `sympy` object.
		# """
		# if not (isinstance(sp_obj, SympyType)):
		# msg = f'Value {sp_obj} is not a `sympy` expression'
		# raise ValueError(msg)

		# return sp_obj

		sympy_expr_schema = pyd_core_schema.chain_schema(
			[
				pyd_core_schema.no_info_plain_validator_function(validate_from_str),
				# pyd_core_schema.no_info_plain_validator_function(validate_from_expr),
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
				'matrix': isinstance(expr, sp.MatrixBase),
			}[allowed_set]
			for allowed_set in allowed_structures
			if allowed_structures != 'scalar'
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
## Technically a "unit expression", which includes compound types.
## Support for this is the killer feature compared to spu.Quantity.
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

# Vector
Real3DVector: typ.TypeAlias = ConstrSympyExpr(
	allow_variables=False,
	allow_units=False,
	allowed_sets={'integer', 'rational', 'real'},
	allowed_structures={'matrix'},
	allowed_matrix_shapes={(3, 1)},
)
