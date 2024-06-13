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

import typing as typ
from fractions import Fraction

import pydantic as pyd
import sympy as sp
import sympy.physics.units as spu
import typing_extensions as typx
from pydantic_core import core_schema as pyd_core_schema

from . import units as spux
from .sympy_type import SympyType
from .unit_analysis import get_units, uses_units


####################
# - Pydantic "Sympy Expr"
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
			return expr.subs(spux.UNIT_BY_SYMBOL)

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


def SympyObj(instance_of: set[typ.Any]) -> typx.Annotated:  # noqa: N802
	"""Declare that a sympy object guaranteed to be an instance of the given bases."""

	def validate_sp_obj(sp_obj: SympyType):
		if any(isinstance(sp_obj, Base) for Base in instance_of):
			return sp_obj

		msg = f'Sympy object {sp_obj} is not an instance of a specified valid base {instance_of}.'
		raise ValueError(msg)

	return typx.Annotated[
		sp.Basic,
		_SympyExpr,
		pyd.AfterValidator(validate_sp_obj),
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
# - Numbers
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
