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

import jax
import jax.numpy as jnp
import sympy as sp

from .. import logger
from .sympy_type import SympyType

log = logger.get(__name__)


####################
# - Parsing: Info from SympyType
####################
def parse_shape(sp_obj: SympyType) -> int | None:
	if isinstance(sp_obj, sp.MatrixBase):
		return sp_obj.shape

	return None


####################
# - Casting: Python
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
# - Casting: Printing
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
	## TODO: A bool flag property that does a lot of find/replace to make it super pretty
	return _SYMPY_EXPR_PRINTER_STR.doprint(sp_obj)


def pretty_symbol(sym: sp.Symbol) -> str:
	return f'{sym.name} ∈ ' + (
		'ℤ'
		if sym.is_integer
		else ('ℝ' if sym.is_real else ('ℂ' if sym.is_complex else '?'))
	)
