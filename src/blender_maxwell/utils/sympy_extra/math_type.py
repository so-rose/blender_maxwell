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

"""Implements `MathType`, a convenient UI-friendly identifier of numerical identity."""

import enum
import sys
import typing as typ
from fractions import Fraction

import jax
import jaxtyping as jtyp
import sympy as sp

from blender_maxwell import contracts as ct

from .. import logger
from .sympy_type import SympyType

log = logger.get(__name__)

MatrixSet: typ.TypeAlias = sp.matrices.MatrixSet


class MathType(enum.StrEnum):
	"""A convenient, UI-friendly identifier of a numerical object's identity."""

	Integer = enum.auto()
	Rational = enum.auto()
	Real = enum.auto()
	Complex = enum.auto()

	####################
	# - Checks
	####################
	@staticmethod
	def has_mathtype(obj: typ.Any) -> typ.Literal['pytype', 'jax', 'expr'] | None:
		"""Determine whether an object of arbitrary type can be considered to have a `MathType`.

		- **Pure Python**: The numerical Python types (`int | Fraction | float | complex`) are all valid.
		- **Expression**: Sympy types / expression are in general considered to have a valid MathType.
		- **Jax**: Non-empty `jax` arrays with a valid numerical Python type as the first element are valid.

		Returns:
			A string literal indicating how to parse the object for a valid `MathType`.

			If the presence of a MathType couldn't be deduced, then return None.
		"""
		if isinstance(obj, int | Fraction | float | complex):
			return 'pytype'

		if (
			isinstance(obj, jax.Array)
			and obj
			and isinstance(obj.item(0), int | Fraction | float | complex)
		):
			return 'jax'

		if isinstance(obj, sp.Basic | sp.MatrixBase):
			return 'expr'
			## TODO: Should we check deeper?

		return None

	####################
	# - Creation
	####################
	@staticmethod
	def from_expr(sp_obj: SympyType, optional: bool = False) -> type | None:  # noqa: PLR0911
		"""Deduce the `MathType` of an arbitrary sympy object (/expression).

		The "assumptions" system of `sympy` is relied on to determine the key properties of the expression.
		To this end, it's important to note several of the shortcomings of the "assumptions" system:

		- All elements, especially symbols, must have well-defined assumptions, ex. `real=True`.
		- Only the "narrowest" possible `MathType` will be deduced, ex. `5` may well be the result of a complex expression, but since it is now an integer, it will parse to `MathType.Integer`. This may break some
		- For infinities, only real and complex infinities are distinguished between in `sympy` (`sp.oo` vs. `sp.zoo`) - aka. there is no "integer infinity" which will parse to `Integer` with this method.

		Warnings:
			Using the "assumptions" system like this requires a lot of rigor in the entire program.

		Notes:
			Any matrix-like object will have `MathType.combine()` run on all of its (flattened) elements.
			This is an extremely **slow** operation, but accurate, according to the semantics of `MathType.combine()`.

			Note that `sp.MatrixSymbol` _cannot have assumptions_, and thus shouldn't be used in `sp_obj`.

		Returns:
			A corresponding `MathType`; else, if `optional=True`, return `None`.

		Raises:
			ValueError: If no corresponding `MathType` could be determined, and `optional=False`.

		"""
		if isinstance(sp_obj, sp.MatrixBase):
			return MathType.combine(
				*[MathType.from_expr(v) for v in sp.flatten(sp_obj)]
			)

		if sp_obj.is_integer:
			return MathType.Integer
		if sp_obj.is_rational:
			return MathType.Rational
		if sp_obj.is_real:
			return MathType.Real
		if sp_obj.is_complex:
			return MathType.Complex

		# Infinities
		if sp_obj in [sp.oo, -sp.oo]:
			return MathType.Real
		if sp_obj in [sp.zoo, -sp.zoo]:
			return MathType.Complex

		if optional:
			return None

		msg = f"Can't determine MathType from sympy object: {sp_obj}"
		raise ValueError(msg)

	@staticmethod
	def from_pytype(dtype: type) -> type:
		"""The `MathType` corresponding to a particular pure-Python type."""
		return {
			int: MathType.Integer,
			Fraction: MathType.Rational,
			float: MathType.Real,
			complex: MathType.Complex,
		}[dtype]

	@staticmethod
	def from_jax_array(data: jtyp.Shaped[jtyp.Array, '...']) -> type:
		"""Deduce the MathType corresponding to a JAX array.

		We go about this by leveraging that:
		- `data` is of a homogeneous type.
		- `data.item(0)` returns a single element of the array w/pure-python type.

		By combing this with `type()` and `MathType.from_pytype`, we can effectively deduce the `MathType` of the entire array with relative efficiency.

		Notes:
			Should also work with numpy arrays.
		"""
		if len(data) > 0:
			return MathType.from_pytype(type(data.item(0)))

		msg = 'Cannot determine MathType from empty jax array.'
		raise ValueError(msg)

	####################
	# - Operations
	####################
	@staticmethod
	def combine(*mathtypes: list[typ.Self], optional: bool = False) -> typ.Self | None:
		if MathType.Complex in mathtypes:
			return MathType.Complex
		if MathType.Real in mathtypes:
			return MathType.Real
		if MathType.Rational in mathtypes:
			return MathType.Rational
		if MathType.Integer in mathtypes:
			return MathType.Integer

		if optional:
			return None

		msg = f"Can't combine mathtypes {mathtypes}"
		raise ValueError(msg)

	def is_compatible(self, other: typ.Self) -> bool:
		MT = MathType
		return (
			other
			in {
				MT.Integer: [MT.Integer],
				MT.Rational: [MT.Integer, MT.Rational],
				MT.Real: [MT.Integer, MT.Rational, MT.Real],
				MT.Complex: [MT.Integer, MT.Rational, MT.Real, MT.Complex],
			}[self]
		)

	def coerce_compatible_pyobj(
		self, pyobj: bool | int | Fraction | float | complex
	) -> int | Fraction | float | complex:
		"""Coerce a pure-python object of numerical type to the _exact_ type indicated by this `MathType`.

		This is needed when ex. one has an integer, but it is important that that integer be passed as a complex number.
		"""
		MT = MathType
		match self:
			case MT.Integer:
				return int(pyobj)
			case MT.Rational if isinstance(pyobj, int):
				return Fraction(pyobj, 1)
			case MT.Rational if isinstance(pyobj, Fraction):
				return pyobj
			case MT.Real:
				return float(pyobj)
			case MT.Complex if isinstance(pyobj, int | Fraction):
				return complex(float(pyobj), 0)
			case MT.Complex if isinstance(pyobj, float):
				return complex(pyobj, 0)

	@staticmethod
	def from_symbolic_set(
		s: sp.Set,
		optional: bool = False,
	) -> typ.Self | None:
		"""Deduce the `MathType` from a particular symbolic set.

		Currently hard-coded.
		Any deviation that might be expected to work, ex. `sp.Reals - {0}`, currently won't (currently).

		Raises:
			ValueError: If a non-hardcoded symbolic set is passed.
		"""
		MT = MathType
		match s:
			case sp.Naturals | sp.Naturals0 | sp.Integers:
				return MT.Integer
			case sp.Rationals:
				return MT.Rational
			case sp.Reals:
				return MT.Real
			case sp.Complexes:
				return MT.Complex

		if isinstance(s, sp.ProductSet):
			return MT.combine(*[MT.from_symbolic_set(arg) for arg in s.sets])

		if isinstance(s, MatrixSet):
			return MT.from_symbolic_set(s.set)

		valid_mathtype = MT.Complex
		for test_set, mathtype in [
			(sp.Complexes, MT.Complex),
			(sp.Reals, MT.Real),
			(sp.Rationals, MT.Rational),
			(sp.Integers, MT.Integer),
		]:
			if s.issubset(test_set):
				valid_mathtype = mathtype
			else:
				return valid_mathtype

		if optional:
			return None

		msg = f"Can't deduce MathType from symbolic set {s}"
		raise ValueError(msg)

	####################
	# - Casting: Pytype
	####################
	@property
	def pytype(self) -> type:
		"""Deduce the pure-Python type that corresponds to this `MathType`."""
		MT = MathType
		return {
			MT.Integer: int,
			MT.Rational: Fraction,
			MT.Real: float,
			MT.Complex: complex,
		}[self]

	@property
	def dtype(self) -> type:
		"""Deduce the type that corresponds to this `MathType`, which is usable with `numpy`/`jax`."""
		MT = MathType
		return {
			MT.Integer: int,
			MT.Rational: float,
			MT.Real: float,
			MT.Complex: complex,
		}[self]

	@property
	def inf_finite(self) -> type:
		"""Opinionated finite representation of "infinity" within this `MathType`.

		These are chosen using `sys.maxsize` and `sys.float_info`.
		As such, while not arbitrary, this "finite representation of infinity" certainly is opinionated.

		**Note** that, in practice, most systems will have no trouble working with values that exceed those defined here.

		Notes:
			Values should be presumed to vary by-platform, as the `sys` attributes may be influenced by CPU architecture, OS, runtime environment, etc. .

			These values can be used directly in `jax` arrays, but at the cost of an overflow warning (in part because `jax` generally only allows the use of `float32`).
			In this case, the warning doesn't matter, as the value will be cast to `jnp.inf` anyway.

			However, it's generally cleaner to directly use `jnp.inf` if infinite values must be defined in an array context.
		"""
		MT = MathType
		Z = MT.Integer
		R = MT.Integer
		return {
			MT.Integer: (-sys.maxsize, sys.maxsize),
			MT.Rational: (
				Fraction(Z.inf_finite[0], 1),
				Fraction(Z.inf_finite[1], 1),
			),
			MT.Real: -(sys.float_info.min, sys.float_info.max),
			MT.Complex: (
				complex(R.inf_finite[0], R.inf_finite[0]),
				complex(R.inf_finite[1], R.inf_finite[1]),
			),
		}[self]

	####################
	# - Casting: Symbolic
	####################
	@property
	def symbolic_set(self) -> sp.Set:
		"""Deduce the symbolic `sp.Set` type that corresponds to this `MathType`."""
		MT = MathType
		return {
			MT.Integer: sp.Integers,
			MT.Rational: sp.Rationals,
			MT.Real: sp.Reals,
			MT.Complex: sp.Complexes,
		}[self]

	@property
	def sp_symbol_a(self) -> type:
		MT = MathType
		return {
			MT.Integer: sp.Symbol('a', integer=True),
			MT.Rational: sp.Symbol('a', rational=True),
			MT.Real: sp.Symbol('a', real=True),
			MT.Complex: sp.Symbol('a', complex=True),
		}[self]

	####################
	# - Labels
	####################
	@staticmethod
	def to_str(value: typ.Self) -> type:
		return {
			MathType.Integer: 'ℤ',
			MathType.Rational: 'ℚ',
			MathType.Real: 'ℝ',
			MathType.Complex: 'ℂ',
		}[value]

	@property
	def name(self) -> str:
		"""Simple non-unicode name of the math type."""
		return str(self)

	@property
	def label_pretty(self) -> str:
		return MathType.to_str(self)

	####################
	# - UI
	####################
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
