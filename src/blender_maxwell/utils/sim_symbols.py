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
import functools
import string
import sys
import typing as typ
from fractions import Fraction

import jaxtyping as jtyp
import pydantic as pyd
import sympy as sp
import sympy.physics.units as spu

from . import extra_sympy_units as spux
from . import logger, serialize

int_min = -(2**64)
int_max = 2**64
float_min = sys.float_info.min
float_max = sys.float_info.max

log = logger.get(__name__)


def unicode_superscript(n: int) -> str:
	"""Transform an integer into its unicode-based superscript character."""
	return ''.join(['⁰¹²³⁴⁵⁶⁷⁸⁹'[ord(c) - ord('0')] for c in str(n)])


####################
# - Simulation Symbol Names
####################
_l = ''
_it_lower = iter(string.ascii_lowercase)


class SimSymbolName(enum.StrEnum):
	# Generic
	Constant = enum.auto()
	Expr = enum.auto()
	Data = enum.auto()

	# Ascii Letters
	while True:
		try:
			globals()['_l'] = next(globals()['_it_lower'])
		except StopIteration:
			break

		locals()[f'Lower{globals()["_l"].upper()}'] = enum.auto()
		locals()[f'Upper{globals()["_l"].upper()}'] = enum.auto()

	# Greek Letters
	LowerTheta = enum.auto()
	LowerPhi = enum.auto()

	# Fields
	Ex = enum.auto()
	Ey = enum.auto()
	Ez = enum.auto()
	Hx = enum.auto()
	Hy = enum.auto()
	Hz = enum.auto()

	Er = enum.auto()
	Etheta = enum.auto()
	Ephi = enum.auto()
	Hr = enum.auto()
	Htheta = enum.auto()
	Hphi = enum.auto()

	# Optics
	Wavelength = enum.auto()
	Frequency = enum.auto()

	PermXX = enum.auto()
	PermYY = enum.auto()
	PermZZ = enum.auto()

	Flux = enum.auto()

	DiffOrderX = enum.auto()
	DiffOrderY = enum.auto()

	####################
	# - UI
	####################
	@staticmethod
	def to_name(v: typ.Self) -> str:
		"""Convert the enum value to a human-friendly name.

		Notes:
			Used to print names in `EnumProperty`s based on this enum.

		Returns:
			A human-friendly name corresponding to the enum value.
		"""
		return SimSymbolName(v).name

	@staticmethod
	def to_icon(_: typ.Self) -> str:
		"""Convert the enum value to a Blender icon.

		Notes:
			Used to print icons in `EnumProperty`s based on this enum.

		Returns:
			A human-friendly name corresponding to the enum value.
		"""
		return ''

	####################
	# - Computed Properties
	####################
	@property
	def name(self) -> str:
		SSN = SimSymbolName
		return (
			# Ascii Letters
			{SSN[f'Lower{letter.upper()}']: letter for letter in string.ascii_lowercase}
			| {
				SSN[f'Upper{letter.upper()}']: letter.upper()
				for letter in string.ascii_lowercase
			}
			| {
				# Generic
				SSN.Constant: 'constant',
				SSN.Expr: 'expr',
				SSN.Data: 'data',
				# Greek Letters
				SSN.LowerTheta: 'theta',
				SSN.LowerPhi: 'phi',
				# Fields
				SSN.Ex: 'Ex',
				SSN.Ey: 'Ey',
				SSN.Ez: 'Ez',
				SSN.Hx: 'Hx',
				SSN.Hy: 'Hy',
				SSN.Hz: 'Hz',
				SSN.Er: 'Ex',
				SSN.Etheta: 'Ey',
				SSN.Ephi: 'Ez',
				SSN.Hr: 'Hx',
				SSN.Htheta: 'Hy',
				SSN.Hphi: 'Hz',
				# Optics
				SSN.Wavelength: 'wl',
				SSN.Frequency: 'freq',
				SSN.PermXX: 'eps_xx',
				SSN.PermYY: 'eps_yy',
				SSN.PermZZ: 'eps_zz',
				SSN.Flux: 'flux',
				SSN.DiffOrderX: 'order_x',
				SSN.DiffOrderY: 'order_y',
			}
		)[self]

	@property
	def name_pretty(self) -> str:
		SSN = SimSymbolName
		return {
			# Generic
			# Greek Letters
			SSN.LowerTheta: 'θ',
			SSN.LowerPhi: 'φ',
			# Fields
			SSN.Etheta: 'Eθ',
			SSN.Ephi: 'Eφ',
			SSN.Hr: 'Hr',
			SSN.Htheta: 'Hθ',
			SSN.Hphi: 'Hφ',
			# Optics
			SSN.Wavelength: 'λ',
			SSN.Frequency: '𝑓',
			SSN.PermXX: 'ε_xx',
			SSN.PermYY: 'ε_yy',
			SSN.PermZZ: 'ε_zz',
		}.get(self, self.name)


####################
# - Simulation Symbol
####################
def mk_interval(
	interval_finite: tuple[int | Fraction | float, int | Fraction | float],
	interval_inf: tuple[bool, bool],
	interval_closed: tuple[bool, bool],
	unit_factor: typ.Literal[1] | spux.Unit,
) -> sp.Interval:
	"""Create a symbolic interval from the tuples (and unit) defining it."""
	return sp.Interval(
		start=(interval_finite[0] * unit_factor if not interval_inf[0] else -sp.oo),
		end=(interval_finite[1] * unit_factor if not interval_inf[1] else sp.oo),
		left_open=(True if interval_inf[0] else not interval_closed[0]),
		right_open=(True if interval_inf[1] else not interval_closed[1]),
	)


class SimSymbol(pyd.BaseModel):
	"""A declarative representation of a symbolic variable.

	`sympy`'s symbols aren't quite flexible enough for our needs: The symbols that we're transporting often need exact domain information, an associated unit dimension, and a great deal of determinism in checks thereof.

	This dataclass is UI-friendly, as it only uses field type annotations/defaults supported by `bl_cache.BLProp`.
	It's easy to persist, easy to transport, and has many helpful properties which greatly simplify working with symbols.
	"""

	model_config = pyd.ConfigDict(frozen=True)

	sym_name: SimSymbolName
	mathtype: spux.MathType = spux.MathType.Real
	physical_type: spux.PhysicalType = spux.PhysicalType.NonPhysical

	# Units
	## -> 'None' indicates that no particular unit has yet been chosen.
	## -> Not exposed in the UI; must be set some other way.
	unit: spux.Unit | None = None
	## -> TODO: We currently allowing units that don't match PhysicalType
	## -> -- In particular, NonPhysical w/units means "unknown units".
	## -> -- This is essential for the Scientific Constant Node.

	# Size
	## -> All SimSymbol sizes are "2D", but interpreted by convention.
	## -> 1x1: "Scalar".
	## -> nx1: "Vector".
	## -> 1xn: "Covector".
	## -> nxn: "Matrix".
	rows: int = 1
	cols: int = 1

	# Scalar Domain: "Interval"
	## -> NOTE: interval_finite_*[0] must be strictly smaller than [1].
	## -> See self.domain.
	## -> We have to deconstruct symbolic interval semantics a bit for UI.
	is_constant: bool = False
	interval_finite_z: tuple[int, int] = (0, 1)
	interval_finite_q: tuple[tuple[int, int], tuple[int, int]] = ((0, 1), (1, 1))
	interval_finite_re: tuple[float, float] = (0.0, 1.0)
	interval_inf: tuple[bool, bool] = (True, True)
	interval_closed: tuple[bool, bool] = (False, False)

	interval_finite_im: tuple[float, float] = (0.0, 1.0)
	interval_inf_im: tuple[bool, bool] = (True, True)
	interval_closed_im: tuple[bool, bool] = (False, False)

	####################
	# - Labels
	####################
	@functools.cached_property
	def name(self) -> str:
		"""Usable name for the symbol."""
		return self.sym_name.name

	@functools.cached_property
	def name_pretty(self) -> str:
		"""Pretty (possibly unicode) name for the thing."""
		return self.sym_name.name_pretty
		## TODO: Formatting conventions for bolding/etc. of vectors/mats/...

	@functools.cached_property
	def mathtype_size_label(self) -> str:
		"""Pretty label that shows both mathtype and size."""
		return f'{self.mathtype.label_pretty}' + (
			'ˣ'.join([unicode_superscript(out_axis) for out_axis in self.shape])
			if self.shape
			else ''
		)

	@functools.cached_property
	def unit_label(self) -> str:
		"""Pretty unit label, which is an empty string when there is no unit."""
		return spux.sp_to_str(self.unit) if self.unit is not None else ''

	@functools.cached_property
	def def_label(self) -> str:
		"""Pretty definition label, exposing the symbol definition."""
		return f'{self.name_pretty} | {self.unit_label} ∈ {self.mathtype_size_label}'
		## TODO: Domain of validity from self.domain?

	@functools.cached_property
	def plot_label(self) -> str:
		"""Pretty plot-oriented label."""
		return f'{self.name_pretty}' + (
			f'({self.unit})' if self.unit is not None else ''
		)

	####################
	# - Computed Properties
	####################
	@functools.cached_property
	def unit_factor(self) -> spux.SympyExpr:
		"""Factor corresponding to the tracked unit, which can be multiplied onto exported values without `None`-checking."""
		return self.unit if self.unit is not None else sp.S(1)

	@functools.cached_property
	def size(self) -> tuple[int, ...] | None:
		return {
			(1, 1): spux.NumberSize1D.Scalar,
			(2, 1): spux.NumberSize1D.Vec2,
			(3, 1): spux.NumberSize1D.Vec3,
			(4, 1): spux.NumberSize1D.Vec4,
		}.get((self.rows, self.cols))

	@functools.cached_property
	def shape(self) -> tuple[int, ...]:
		match (self.rows, self.cols):
			case (1, 1):
				return ()
			case (_, 1):
				return (self.rows,)
			case (1, _):
				return (1, self.rows)
			case (_, _):
				return (self.rows, self.cols)

	@functools.cached_property
	def shape_len(self) -> spux.SympyExpr:
		"""Factor corresponding to the tracked unit, which can be multiplied onto exported values without `None`-checking."""
		return len(self.shape)

	@functools.cached_property
	def domain(self) -> sp.Interval | sp.Set:
		"""Return the scalar domain of valid values for each element of the symbol.

		For integer/rational/real symbols, the domain is an interval defined using the `interval_*` properties.
		This interval **must** have the property`start <= stop`.

		Otherwise, the domain is the symbolic set corresponding to `self.mathtype`.
		"""
		match self.mathtype:
			case spux.MathType.Integer:
				return mk_interval(
					self.interval_finite_z,
					self.interval_inf,
					self.interval_closed,
					self.unit_factor,
				)

			case spux.MathType.Rational:
				return mk_interval(
					Fraction(*self.interval_finite_q),
					self.interval_inf,
					self.interval_closed,
					self.unit_factor,
				)

			case spux.MathType.Real:
				return mk_interval(
					self.interval_finite_re,
					self.interval_inf,
					self.interval_closed,
					self.unit_factor,
				)

			case spux.MathType.Complex:
				return (
					mk_interval(
						self.interval_finite_re,
						self.interval_inf,
						self.interval_closed,
						self.unit_factor,
					),
					mk_interval(
						self.interval_finite_im,
						self.interval_inf_im,
						self.interval_closed_im,
						self.unit_factor,
					),
				)

	@functools.cached_property
	def valid_domain_value(self) -> spux.SympyExpr:
		"""A single value guaranteed to be conformant to this `SimSymbol` and within `self.domain`."""
		match (self.domain.start.is_finite, self.domain.end.is_finite):
			case (True, True):
				if self.mathtype is spux.MathType.Integer:
					return (self.domain.start + self.domain.end) // 2
				return (self.domain.start + self.domain.end) / 2

			case (True, False):
				one = sp.S(self.mathtype.coerce_compatible_pyobj(-1))
				return self.domain.start + one

			case (False, True):
				one = sp.S(self.mathtype.coerce_compatible_pyobj(-1))
				return self.domain.end - one

			case (False, False):
				return sp.S(self.mathtype.coerce_compatible_pyobj(-1))

	####################
	# - Properties
	####################
	@functools.cached_property
	def sp_symbol(self) -> sp.Symbol | sp.ImmutableMatrix:
		"""Return a symbolic variable w/unit, corresponding to this `SimSymbol`.

		As much as possible, appropriate `assumptions` are set in the constructor of `sp.Symbol`, insofar as they can be determined.

		- **MathType**: Depending on `self.mathtype`.
		- **Positive/Negative**: Depending on `self.domain`.
		- **Nonzero**: Depending on `self.domain`, including open/closed boundary specifications.

		Notes:
			**The assumptions system is rather limited**, and implementations should strongly consider transporting `SimSymbols` instead of `sp.Symbol`.

			This allows tracking ex. the valid interval domain for a symbol.
		"""
		# MathType Assumption
		mathtype_kwargs = {}
		match self.mathtype:
			case spux.MathType.Integer:
				mathtype_kwargs |= {'integer': True}
			case spux.MathType.Rational:
				mathtype_kwargs |= {'rational': True}
			case spux.MathType.Real:
				mathtype_kwargs |= {'real': True}
			case spux.MathType.Complex:
				mathtype_kwargs |= {'complex': True}

		# Non-Zero Assumption
		if (
			(
				self.domain.left == 0
				and self.domain.left_open
				or self.domain.right == 0
				and self.domain.right_open
			)
			or self.domain.left > 0
			or self.domain.right < 0
		):
			mathtype_kwargs |= {'nonzero': True}

		# Positive/Negative Assumption
		if self.domain.left >= 0:
			mathtype_kwargs |= {'positive': True}
		elif self.domain.right <= 0:
			mathtype_kwargs |= {'negative': True}

		# Scalar: Return Symbol
		if self.rows == 1 and self.cols == 1:
			return sp.Symbol(self.sym_name.name, **mathtype_kwargs)

		# Vector|Matrix: Return Matrix of Symbols
		## -> MatrixSymbol doesn't support assumptions.
		## -> This little construction does.
		return sp.ImmutableMatrix(
			[
				[
					sp.Symbol(self.sym_name.name + f'_{row}{col}', **mathtype_kwargs)
					for col in range(self.cols)
				]
				for row in range(self.rows)
			]
		)

	@functools.cached_property
	def sp_symbol_matsym(self) -> sp.Symbol | sp.MatrixSymbol:
		"""Return a symbolic variable w/unit, corresponding to this `SimSymbol`, w/variable shape support.

		To preserve as many assumptions as possible, `self.sp_symbol` returns a matrix of individual `sp.Symbol`s whenever the `SimSymbol` is non-scalar.
		However, this isn't always the most useful representation: For example, if the intention is to use a shaped symbolic variable as an argument to `sympy.lambdify()`, one would have to flatten each individual `sp.Symbol` and pass each matrix element as a single element, greatly complicating things like broadcasting.

		For this reason, this property is provided.
		Whenever the `SimSymbol` is scalar, it works identically to `self.sp_symbol`.
		However, when the `SimSymbol` is shaped, an appropriate `sp.MatrixSymbol` is returned instead.

		Notes:
			`sp.MatrixSymbol` doesn't support assumptions.
			As such, things like deduction of `MathType` from expressions involving a matrix symbol simply won't work.
		"""
		if self.shape_len == 0:
			return self.sp_symbol
		return sp.MatrixSymbol(self.sym_name.name, self.rows, self.cols)

	@functools.cached_property
	def sp_symbol_phy(self) -> spux.SympyExpr:
		"""Physical symbol containing `self.sp_symbol` multiplied by `self.unit`."""
		return self.sp_symbol * self.unit_factor

	@functools.cached_property
	def expr_info(self) -> dict[str, typ.Any]:
		"""Generate keyword arguments for an ExprSocket, whose output values will be guaranteed to conform to this `SimSymbol`.

		Notes:
			Before use, `active_kind=ct.FlowKind.Range` can be added to make the `ExprSocket`.

			Default values are set for both `Value` and `Range`.
			To this end, `self.domain` is used.

			Since `ExprSocketDef` allows the use of infinite bounds for `default_min` and `default_max`, we defer the decision of how to treat finite-fallback to the `ExprSocketDef`.
		"""
		if self.size is not None:
			if self.unit in self.physical_type.valid_units:
				return {
					'output_name': self.sym_name,
					# Socket Interface
					'size': self.size,
					'mathtype': self.mathtype,
					'physical_type': self.physical_type,
					# Defaults: Units
					'default_unit': self.unit,
					'default_symbols': [],
					# Defaults: FlowKind.Value
					'default_value': self.conform(
						self.valid_domain_value, strip_unit=True
					),
					# Defaults: FlowKind.Range
					'default_min': self.domain.start,
					'default_max': self.domain.end,
				}
			msg = f'Tried to generate an ExprSocket from a SymSymbol "{self.name}", but its unit ({self.unit}) is not a valid unit of its physical type ({self.physical_type}) (SimSymbol={self})'
			raise NotImplementedError(msg)
		msg = f'Tried to generate an ExprSocket from a SymSymbol "{self.name}", but its size ({self.rows} by {self.cols}) is incompatible with ExprSocket (SimSymbol={self})'
		raise NotImplementedError(msg)

	####################
	# - Operations
	####################
	def update(self, **kwargs) -> typ.Self:
		def get_attr(attr: str):
			_notfound = 'notfound'
			if kwargs.get(attr, _notfound) is _notfound:
				return getattr(self, attr)
			return kwargs[attr]

		return SimSymbol(
			sym_name=get_attr('sym_name'),
			mathtype=get_attr('mathtype'),
			physical_type=get_attr('physical_type'),
			unit=get_attr('unit'),
			rows=get_attr('rows'),
			cols=get_attr('cols'),
			interval_finite_z=get_attr('interval_finite_z'),
			interval_finite_q=get_attr('interval_finite_q'),
			interval_finite_re=get_attr('interval_finite_re'),
			interval_inf=get_attr('interval_inf'),
			interval_closed=get_attr('interval_closed'),
			interval_finite_im=get_attr('interval_finite_im'),
			interval_inf_im=get_attr('interval_inf_im'),
			interval_closed_im=get_attr('interval_closed_im'),
		)

	def set_finite_domain(  # noqa: PLR0913
		self,
		start: int | float,
		end: int | float,
		start_closed: bool = True,
		end_closed: bool = True,
		start_im: bool = float,
		end_im: bool = float,
		start_closed_im: bool = True,
		end_closed_im: bool = True,
	) -> typ.Self:
		"""Update the symbol with a finite range."""
		closed_re = (start_closed, end_closed)
		closed_im = (start_closed_im, end_closed_im)
		match self.mathtype:
			case spux.MathType.Integer:
				return self.update(
					interval_finite_z=(start, end),
					interval_inf=(False, False),
					interval_closed=closed_re,
				)
			case spux.MathType.Rational:
				return self.update(
					interval_finite_q=(start, end),
					interval_inf=(False, False),
					interval_closed=closed_re,
				)
			case spux.MathType.Real:
				return self.update(
					interval_finite_re=(start, end),
					interval_inf=(False, False),
					interval_closed=closed_re,
				)
			case spux.MathType.Complex:
				return self.update(
					interval_finite_re=(start, end),
					interval_finite_im=(start_im, end_im),
					interval_inf=(False, False),
					interval_closed=closed_re,
					interval_closed_im=closed_im,
				)

	def set_size(self, rows: int, cols: int) -> typ.Self:
		return self.update(rows=rows, cols=cols)

	def conform(
		self, sp_obj: spux.SympyType, strip_unit: bool = False
	) -> spux.SympyType:
		"""Conform a sympy object to the properties of this `SimSymbol`, if possible.

		To achieve this, a number of operations may be performed:

		- **Unit Conversion**: If the object has no units, but should, multiply by `self.unit`. If the object has units, but shouldn't, strip them. Otherwise, convert its unit to `self.unit`.
		- **Broadcast Expansion**: If the object is a scalar, but the `SimSymbol` is shaped, then an `sp.ImmutableMatrix` is returned with the scalar at each position.

		Returns:
			A transformed sympy object guaranteed usable as a particular value of this `SimSymbol` variable.

		Raises:
			ValueError: If the units of `sp_obj` can't be cleanly converted to `self.unit`.
		"""
		res = sp_obj

		# Unit Conversion
		match (spux.uses_units(sp_obj), self.unit is not None):
			case (True, True):
				res = spux.scale_to_unit(sp_obj, self.unit) * self.unit

			case (False, True):
				res = sp_obj * self.unit

			case (True, False):
				res = spux.strip_unit_system(sp_obj)

		if strip_unit:
			res = spux.strip_unit_system(sp_obj)

		# Broadcast Expansion
		if self.rows > 1 or self.cols > 1 and not isinstance(res, spux.MatrixBase):
			res = sp_obj * sp.ImmutableMatrix.ones(self.rows, self.cols)

		return res

	def scale(
		self, sp_obj: spux.SympyType, use_jax_array: bool = True
	) -> int | float | complex | jtyp.Inexact[jtyp.Array, '...']:
		"""Remove all symbolic elements from the conformed `sp_obj`, preparing it for use in contexts that don't support unrealized symbols.

		On top of `self.conform()`, a number of operations are performed.

		- **Unit Stripping**: The `self.unit` of the expression returned by `self.conform()` will be stripped.
		- **Sympy to Python**: The now symbol-less expression will be converted to either a pure Python type, or to a `jax` array (if `use_jax_array` is set).

		Notes:
			When creating numerical functions of expressions using `.lambdify`, `self.scale()` **must be used** in place of `self.conform()` before the parameterized expression is used.

		Returns:
			A "raw" (pure Python / jax array) type guaranteed usable as a particular **numerical** value of this `SymSymbol` variable.
		"""
		# Conform
		res = self.conform(sp_obj)

		# Strip Units
		res = spux.scale_to_unit(sp_obj, self.unit)

		# Sympy to Python
		res = spux.sympy_to_python(res, use_jax_array=use_jax_array)

		return res  # noqa: RET504

	@staticmethod
	def from_expr(
		sym_name: SimSymbolName,
		expr: spux.SympyExpr,
		unit_expr: spux.SympyExpr,
	) -> typ.Self:
		"""Deduce a `SimSymbol` that matches the output of a given expression (and unit expression).

		This is an essential method, allowing for the ded

		Notes:
			`PhysicalType` **cannot be set** from an expression in the generic sense.
			Therefore, the trick of using `NonPhysical` with non-`None` unit to denote unknown `PhysicalType` is used in the output.

			All intervals are kept at their defaults.

		Parameters:
			sym_name: The `SimSymbolName` to set to the resulting symbol.
			expr: The unit-aware expression to parse and encapsulate as a symbol.
			unit_expr: A dimensional analysis expression (set to `1` to make the resulting symbol unitless).
				Fundamentally, units are just the variables of scalar terms.
				'1' for unitless terms are, in the dimanyl sense, constants.

				Doing it like this may be a little messy, but is accurate.

		Returns:
			A fresh new `SimSymbol` that tries to match the given expression (and unit expression) well enough to be usable in place of it.
		"""
		# MathType from Expr Assumptions
		## -> All input symbols have assumptions, because we are very pedantic.
		## -> Therefore, we should be able to reconstruct the MathType.
		mathtype = spux.MathType.from_expr(expr)

		# PhysicalType as "NonPhysical"
		## -> 'unit' still applies - but we can't guarantee a PhysicalType will.
		## -> Therefore, this is what we gotta do.
		physical_type = spux.PhysicalType.NonPhysical

		# Rows/Cols from Expr (if Matrix)
		rows, cols = expr.shape if isinstance(expr, sp.MatrixBase) else (1, 1)

		return SimSymbol(
			sym_name=sym_name,
			mathtype=mathtype,
			physical_type=physical_type,
			unit=unit_expr if unit_expr != 1 else None,
			rows=rows,
			cols=cols,
		)

	####################
	# - Serialization
	####################
	def dump_as_msgspec(self) -> serialize.NaiveRepresentation:
		"""Transforms this `SimSymbol` into an object that can be natively serialized by `msgspec`.

		Notes:
			Makes use of `pydantic.BaseModel.model_dump()` to cast any special fields into a serializable format.
			If this method is failing, check that `pydantic` can actually cast all the fields in your model.

		Returns:
			A particular `list`, with two elements:

			1. The `serialize`-provided "Type Identifier", to differentiate this list from generic list.
			2. A dictionary containing simple Python types, as cast by `pydantic`.
		"""
		return [serialize.TypeID.SimSymbol, self.__class__.__name__, self.model_dump()]

	@staticmethod
	def parse_as_msgspec(obj: serialize.NaiveRepresentation) -> typ.Self:
		"""Transforms an object made by `self.dump_as_msgspec()` into an instance of `SimSymbol`.

		Notes:
			The method presumes that the deserialized object produced by `msgspec` perfectly matches the object originally created by `self.dump_as_msgspec()`.

			This is a **mostly robust** presumption, as `pydantic` attempts to be quite consistent in how to interpret types with almost identical semantics.
			Still, yet-unknown edge cases may challenge these presumptions.

		Returns:
			A new instance of `SimSymbol`, initialized using the `model_dump()` dictionary.
		"""
		return SimSymbol(**obj[2])


####################
# - Common Sim Symbols
####################
class CommonSimSymbol(enum.StrEnum):
	"""Identifiers for commonly used `SimSymbol`s, with all information about ex. `MathType`, `PhysicalType`, and (in general) valid intervals all pre-loaded.

	The enum is UI-compatible making it easy to declare a UI-driven dropdown of commonly used symbols that will all behave as expected.

	Attributes:
		X:
		Time: A symbol representing a real-valued wavelength.
		Wavelength: A symbol representing a real-valued wavelength.
			Implicitly, this symbol often represents "vacuum wavelength" in particular.
		Wavelength: A symbol representing a real-valued frequency.
			Generally, this is the non-angular frequency.
	"""

	Index = enum.auto()

	# Space|Time
	SpaceX = enum.auto()
	SpaceY = enum.auto()
	SpaceZ = enum.auto()

	AngR = enum.auto()
	AngTheta = enum.auto()
	AngPhi = enum.auto()

	DirX = enum.auto()
	DirY = enum.auto()
	DirZ = enum.auto()

	Time = enum.auto()

	# Fields
	FieldEx = enum.auto()
	FieldEy = enum.auto()
	FieldEz = enum.auto()
	FieldHx = enum.auto()
	FieldHy = enum.auto()
	FieldHz = enum.auto()

	FieldEr = enum.auto()
	FieldEtheta = enum.auto()
	FieldEphi = enum.auto()
	FieldHr = enum.auto()
	FieldHtheta = enum.auto()
	FieldHphi = enum.auto()

	# Optics
	Wavelength = enum.auto()
	Frequency = enum.auto()

	Flux = enum.auto()

	DiffOrderX = enum.auto()
	DiffOrderY = enum.auto()

	####################
	# - UI
	####################
	@staticmethod
	def to_name(v: typ.Self) -> str:
		"""Convert the enum value to a human-friendly name.

		Notes:
			Used to print names in `EnumProperty`s based on this enum.

		Returns:
			A human-friendly name corresponding to the enum value.
		"""
		return CommonSimSymbol(v).name

	@staticmethod
	def to_icon(_: typ.Self) -> str:
		"""Convert the enum value to a Blender icon.

		Notes:
			Used to print icons in `EnumProperty`s based on this enum.

		Returns:
			A human-friendly name corresponding to the enum value.
		"""
		return ''

	####################
	# - Properties
	####################
	@property
	def name(self) -> str:
		SSN = SimSymbolName
		CSS = CommonSimSymbol
		return {
			CSS.Index: SSN.LowerI,
			# Space|Time
			CSS.SpaceX: SSN.LowerX,
			CSS.SpaceY: SSN.LowerY,
			CSS.SpaceZ: SSN.LowerZ,
			CSS.AngR: SSN.LowerR,
			CSS.AngTheta: SSN.LowerTheta,
			CSS.AngPhi: SSN.LowerPhi,
			CSS.DirX: SSN.LowerX,
			CSS.DirY: SSN.LowerY,
			CSS.DirZ: SSN.LowerZ,
			CSS.Time: SSN.LowerT,
			# Fields
			CSS.FieldEx: SSN.Ex,
			CSS.FieldEy: SSN.Ey,
			CSS.FieldEz: SSN.Ez,
			CSS.FieldHx: SSN.Hx,
			CSS.FieldHy: SSN.Hy,
			CSS.FieldHz: SSN.Hz,
			CSS.FieldEr: SSN.Er,
			CSS.FieldHr: SSN.Hr,
			# Optics
			CSS.Frequency: SSN.Frequency,
			CSS.Wavelength: SSN.Wavelength,
			CSS.DiffOrderX: SSN.DiffOrderX,
			CSS.DiffOrderY: SSN.DiffOrderY,
		}[self]

	def sim_symbol(self, unit: spux.Unit | None) -> SimSymbol:
		"""Retrieve the `SimSymbol` associated with the `CommonSimSymbol`."""
		CSS = CommonSimSymbol

		# Space
		sym_space = SimSymbol(
			sym_name=self.name,
			physical_type=spux.PhysicalType.Length,
			unit=unit,
		)
		sym_ang = SimSymbol(
			sym_name=self.name,
			physical_type=spux.PhysicalType.Angle,
			unit=unit,
		)

		# Fields
		def sym_field(eh: typ.Literal['e', 'h']) -> SimSymbol:
			return SimSymbol(
				sym_name=self.name,
				physical_type=spux.PhysicalType.EField
				if eh == 'e'
				else spux.PhysicalType.HField,
				unit=unit,
				interval_finite_re=(0, float_max),
				interval_inf_re=(False, True),
				interval_closed_re=(True, False),
				interval_finite_im=(float_min, float_max),
				interval_inf_im=(True, True),
			)

		return {
			CSS.Index: SimSymbol(
				sym_name=self.name,
				mathtype=spux.MathType.Integer,
				interval_finite_z=(0, 2**64),
				interval_inf=(False, True),
				interval_closed=(True, False),
			),
			# Space|Time
			CSS.SpaceX: sym_space,
			CSS.SpaceY: sym_space,
			CSS.SpaceZ: sym_space,
			CSS.AngR: sym_space,
			CSS.AngTheta: sym_ang,
			CSS.AngPhi: sym_ang,
			CSS.Time: SimSymbol(
				sym_name=self.name,
				physical_type=spux.PhysicalType.Time,
				unit=unit,
				interval_finite_re=(0, float_max),
				interval_inf=(False, True),
				interval_closed=(True, False),
			),
			# Fields
			CSS.FieldEx: sym_field('e'),
			CSS.FieldEy: sym_field('e'),
			CSS.FieldEz: sym_field('e'),
			CSS.FieldHx: sym_field('h'),
			CSS.FieldHy: sym_field('h'),
			CSS.FieldHz: sym_field('h'),
			CSS.FieldEr: sym_field('e'),
			CSS.FieldEtheta: sym_field('e'),
			CSS.FieldEphi: sym_field('e'),
			CSS.FieldHr: sym_field('h'),
			CSS.FieldHtheta: sym_field('h'),
			CSS.FieldHphi: sym_field('h'),
			# Optics
			CSS.Wavelength: SimSymbol(
				sym_name=self.name,
				mathtype=spux.MathType.Real,
				physical_type=spux.PhysicalType.Length,
				unit=unit,
				interval_finite=(0, float_max),
				interval_inf=(False, True),
				interval_closed=(False, False),
			),
			CSS.Frequency: SimSymbol(
				sym_name=self.name,
				mathtype=spux.MathType.Real,
				physical_type=spux.PhysicalType.Freq,
				unit=unit,
				interval_finite=(0, float_max),
				interval_inf=(False, True),
				interval_closed=(False, False),
			),
			CSS.Flux: SimSymbol(
				sym_name=SimSymbolName.Flux,
				mathtype=spux.MathType.Real,
				physical_type=spux.PhysicalType.Power,
				unit=unit,
			),
			CSS.DiffOrderX: SimSymbol(
				sym_name=self.name,
				mathtype=spux.MathType.Integer,
				interval_finite=(int_min, int_max),
				interval_inf=(True, True),
				interval_closed=(False, False),
			),
			CSS.DiffOrderY: SimSymbol(
				sym_name=self.name,
				mathtype=spux.MathType.Integer,
				interval_finite=(int_min, int_max),
				interval_inf=(True, True),
				interval_closed=(False, False),
			),
		}[self]


####################
# - Selected Direct-Access to SimSymbols
####################
idx = CommonSimSymbol.Index.sim_symbol
t = CommonSimSymbol.Time.sim_symbol
wl = CommonSimSymbol.Wavelength.sim_symbol
freq = CommonSimSymbol.Frequency.sim_symbol

space_x = CommonSimSymbol.SpaceX.sim_symbol
space_y = CommonSimSymbol.SpaceY.sim_symbol
space_z = CommonSimSymbol.SpaceZ.sim_symbol

dir_x = CommonSimSymbol.DirX.sim_symbol
dir_y = CommonSimSymbol.DirY.sim_symbol
dir_z = CommonSimSymbol.DirZ.sim_symbol

ang_r = CommonSimSymbol.AngR.sim_symbol
ang_theta = CommonSimSymbol.AngTheta.sim_symbol
ang_phi = CommonSimSymbol.AngPhi.sim_symbol

field_ex = CommonSimSymbol.FieldEx.sim_symbol
field_ey = CommonSimSymbol.FieldEy.sim_symbol
field_ez = CommonSimSymbol.FieldEz.sim_symbol
field_hx = CommonSimSymbol.FieldHx.sim_symbol
field_hy = CommonSimSymbol.FieldHx.sim_symbol
field_hz = CommonSimSymbol.FieldHx.sim_symbol

flux = CommonSimSymbol.Flux.sim_symbol

diff_order_x = CommonSimSymbol.DiffOrderX.sim_symbol
diff_order_y = CommonSimSymbol.DiffOrderY.sim_symbol
