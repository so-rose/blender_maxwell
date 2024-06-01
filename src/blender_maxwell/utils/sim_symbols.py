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

from . import logger, serialize
from . import sympy_extra as spux

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

	# EM Fields
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

	Perm = enum.auto()
	PermXX = enum.auto()
	PermYY = enum.auto()
	PermZZ = enum.auto()

	Flux = enum.auto()

	DiffOrderX = enum.auto()
	DiffOrderY = enum.auto()

	BlochX = enum.auto()
	BlochY = enum.auto()
	BlochZ = enum.auto()

	# New Backwards Compatible Entries
	## -> Ordered lists carry a particular enum integer index.
	## -> Therefore, anything but adding an index breaks backwards compat.
	## -> ...With all previous files.
	ConstantRange = enum.auto()

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
				SSN.Constant: 'cst',
				SSN.ConstantRange: 'cst_range',
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
				SSN.Perm: 'eps_r',
				SSN.PermXX: 'eps_xx',
				SSN.PermYY: 'eps_yy',
				SSN.PermZZ: 'eps_zz',
				SSN.Flux: 'flux',
				SSN.DiffOrderX: 'order_x',
				SSN.DiffOrderY: 'order_y',
				SSN.BlochX: 'bloch_x',
				SSN.BlochY: 'bloch_y',
				SSN.BlochZ: 'bloch_z',
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
			SSN.Er: 'Er',
			SSN.Etheta: 'Eθ',
			SSN.Ephi: 'Eφ',
			SSN.Hr: 'Hr',
			SSN.Htheta: 'Hθ',
			SSN.Hphi: 'Hφ',
			# Optics
			SSN.Wavelength: 'λ',
			SSN.Frequency: 'fᵣ',
			SSN.Perm: 'εᵣ',
			SSN.PermXX: 'εᵣ[xx]',
			SSN.PermYY: 'εᵣ[yy]',
			SSN.PermZZ: 'εᵣ[zz]',
		}.get(self, self.name)


####################
# - Simulation Symbol
####################
def mk_interval(
	interval_finite: tuple[int | Fraction | float, int | Fraction | float],
	interval_inf: tuple[bool, bool],
	interval_closed: tuple[bool, bool],
) -> sp.Interval:
	"""Create a symbolic interval from the tuples (and unit) defining it."""
	return sp.Interval(
		start=(interval_finite[0] if not interval_inf[0] else -sp.oo),
		end=(interval_finite[1] if not interval_inf[1] else sp.oo),
		left_open=(True if interval_inf[0] else not interval_closed[0]),
		right_open=(True if interval_inf[1] else not interval_closed[1]),
	)


class SimSymbol(pyd.BaseModel):
	"""A convenient, constrained representation of a symbolic variable suitable for many tasks.

	The original motivation was to enhance `sp.Symbol` with greater flexibility, semantic context, and a UI-friendly representation.
	Today, `SimSymbol` is a fully capable primitive for defining the interfaces between externally tracked mathematical elements, and planning the required operations between them.

	A symbol represented as `SimSymbol` carries all the semantic meaning of that symbol, and comes with a comprehensive library of useful (computed) properties and methods.
	It is immutable, hashable, and serializable, and as a `pydantic.BaseModel` with aggressive property caching, its performance properties should also be well-suited for use in the hot-loops of ex. UI draw methods.

	Attributes:
		sym_name: For humans and computers, symbol names induces a lot of implicit semantics.
		mathtype: Symbols are associated with some set of valid values.
			We choose to constrain `SimSymbol` to only associate with _mathematical_ (aka. number-like) sets.
			This prohibits ex. booleans and predicate-logic applications, but eases a lot of burdens associated with actually using `SimSymbol`.
		physical_type: Symbols may be associated with a particular unit dimension expression.
			This allows the symbol to have _physical meaning_.
			This information is **generally not** encoded in auxiliary attributes like `self.domain`, but **generally is** encoded by computed properties/methods.
		unit: Symbols may be associated with a particular unit, which must be compatible with the `PhysicalType`.
			**NOTE**: Unit expressions may well have physical meaning, without being strictly conformable to a pre-blessed `PhysicalType`s.
			We do try to avoid such cases, but for the sake of correctness, our chosen convention is to let `self.physical_type` be "`NonPhysical`", while still allowing a unit.
		size: Symbols may themselves have shape.
			**NOTE**: We deliberately choose to constrain `SimSymbol`s to two dimensions, allowing them to represent scalars, vectors, covectors, and matrices, but **not** arbitrary tensors.
			This is a practical tradeoff, made both to make it easier (in terms of mathematical analysis) to implement `SimSymbol`, but also to make it easier to define UI elements that drive / are driven by `SimSymbol`s.
		domain: Symbols are associated with a _domain of valid values_, expressed with any mathematical set implemented as a subclass of `sympy.Set`.
			By using a true symbolic set, we gain unbounded flexibility in how to define the validity of a set, including an extremely capable `* in self.domain` operator encapsulating a lot of otherwise very manual logic.
			**NOTE** that `self.unit` is **not** baked into the domain, due to practicalities associated with subclasses of `sp.Set`.

	"""

	model_config = pyd.ConfigDict(frozen=True)

	sym_name: SimSymbolName
	mathtype: spux.MathType = spux.MathType.Real
	physical_type: spux.PhysicalType = spux.PhysicalType.NonPhysical

	# Units
	## -> 'None' indicates that no particular unit has yet been chosen.
	## -> When 'self.physical_type' is NonPhysical, can only be None.
	unit: spux.Unit | None = None

	# Size
	## -> All SimSymbol sizes are "2D", but interpreted by convention.
	## -> 1x1: "Scalar".
	## -> nx1: "Vector".
	## -> 1xn: "Covector".
	## -> nxn: "Matrix".
	rows: int = 1
	cols: int = 1

	# Valid Domain
	## -> Declares the valid set of values that may be given to this symbol.
	## -> By convention, units are not encoded in the domain sp.Set.
	## -> 'sp.Set's are extremely expressive and cool.
	domain: spux.SympyExpr | None = None

	@functools.cached_property
	def domain_mat(self) -> sp.Set | sp.matrices.MatrixSet:
		if self.rows > 1 or self.cols > 1:
			return sp.matrices.MatrixSet(self.rows, self.cols, self.domain)
		return self.domain

	preview_value: spux.SympyExpr | None = None

	####################
	# - Validators
	####################
	## TODO: Check domain against MathType
	## -- Surprisingly hard without a lot of special-casing.

	## TODO: Check that size is valid for the PhysicalType.

	## TODO: Check that constant value (domain=FiniteSet(cst)) is compatible with the MathType.

	## TODO: Check that preview_value is in the domain.

	@pyd.model_validator(mode='after')
	def set_undefined_domain_from_mathtype(self) -> typ.Self:
		"""When the domain is not set, then set it using the symbolic set of the MathType."""
		if self.domain is None:
			object.__setattr__(self, 'domain', self.mathtype.symbolic_set)
		return self

	@pyd.model_validator(mode='after')
	def conform_undefined_preview_value_to_constant(self) -> typ.Self:
		"""When the `SimSymbol` is a constant, but the preview value is not set, then set the preview value from the constant."""
		if self.is_constant and not self.preview_value:
			object.__setattr__(self, 'preview_value', self.constant_value)
		return self

	@pyd.model_validator(mode='after')
	def conform_preview_value(self) -> typ.Self:
		"""Conform the given preview value to the `SimSymbol`."""
		if self.is_constant and not self.preview_value:
			object.__setattr__(
				self,
				'preview_value',
				self.conform(self.preview_value, strip_units=True),
			)
		return self

	####################
	# - Domain
	####################
	@functools.cached_property
	def is_constant(self) -> bool:
		"""When the symbol domain is a single-element `sp.FiniteSet`, then the symbol can be considered to be a constant."""
		return isinstance(self.domain, sp.FiniteSet) and len(self.domain) == 1

	@functools.cached_property
	def constant_value(self) -> bool:
		"""Get the constant when `is_constant` is True.

		The `self.unit_factor` is multiplied onto the constant at this point.
		"""
		if self.is_constant:
			return next(iter(self.domain)) * self.unit_factor

		msg = 'Tried to get constant value of non-constant SimSymbol.'
		raise ValueError(msg)

	@functools.cached_property
	def is_nonzero(self) -> bool:
		"""Whether $0$ is a valid value for this symbol.

		When shaped, $0$ refers to the relevant shaped object with all elements $0$.

		Notes:
			Most notably, this symbol cannot be used as the right hand side of a division operation when this property is `False`.
		"""
		return 0 in self.domain

	####################
	# - Labels
	####################
	@functools.cached_property
	def name(self) -> str:
		"""Usable string name for the symbol."""
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
		return spux.sp_to_str(self.unit.n(4)) if self.unit is not None else ''

	@functools.cached_property
	def name_unit_label(self) -> str:
		"""Pretty name | unit label, which is just the name when there is no unit."""
		if self.unit is None:
			return self.name_pretty
		return f'{self.name_pretty} | {self.unit_label}'

	@functools.cached_property
	def def_label(self) -> str:
		"""Pretty definition label, exposing the symbol definition."""
		return f'{self.name_unit_label} ∈ {self.mathtype_size_label}'
		## TODO: Domain of validity from self.domain?

	@functools.cached_property
	def plot_label(self) -> str:
		"""Pretty plot-oriented label."""
		if self.unit is None:
			return self.name_pretty
		return f'{self.name_pretty} ({self.unit_label})'

	####################
	# - Computed Properties
	####################
	@functools.cached_property
	def unit_factor(self) -> spux.SympyExpr:
		"""Factor corresponding to the tracked unit, which can be multiplied onto exported values without `None`-checking."""
		return self.unit if self.unit is not None else sp.S(1)

	@functools.cached_property
	def unit_dim(self) -> spux.SympyExpr:
		"""Unit dimension factor corresponding to the tracked unit, which can be multiplied onto exported values without `None`-checking."""
		return self.unit if self.unit is not None else sp.S(1)

	@functools.cached_property
	def size(self) -> spux.NumberSize1D | None:
		"""The 1D number size of this `SimSymbol`, if it has one; else None."""
		return {
			(1, 1): spux.NumberSize1D.Scalar,
			(2, 1): spux.NumberSize1D.Vec2,
			(3, 1): spux.NumberSize1D.Vec3,
			(4, 1): spux.NumberSize1D.Vec4,
		}.get((self.rows, self.cols))

	@functools.cached_property
	def shape(self) -> tuple[int, ...]:
		"""Deterministic chosen shape of this `SimSymbol`.

		Derived from `self.rows` and `self.cols`.

		Is never `None`; instead, empty tuple `()` is used.
		"""
		match (self.rows, self.cols):
			case (1, 1):
				return ()
			case (_, 1):
				return (self.rows,)
			case (_, _):
				return (self.rows, self.cols)

	@functools.cached_property
	def shape_len(self) -> spux.SympyExpr:
		"""Factor corresponding to the tracked unit, which can be multiplied onto exported values without `None`-checking."""
		return len(self.shape)

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
		if self.is_nonzero:
			mathtype_kwargs |= {'nonzero': True}

		# Positive/Negative Assumption
		if self.mathtype is not spux.MathType.Complex:
			if self.domain.inf >= 0:
				mathtype_kwargs |= {'positive': True}
			elif self.domain.sup < 0:
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
				socket_info = {
					'output_name': self.sym_name,
					# Socket Interface
					'size': self.size,
					'mathtype': self.mathtype,
					'physical_type': self.physical_type,
					# Defaults: Units
					'default_unit': self.unit,
					'default_symbols': [],
				}

				# Defaults: FlowKind.Value
				if self.preview_value:
					socket_info |= {
						'default_value': self.conform(
							self.preview_value, strip_unit=True
						)
					}

				# Defaults: FlowKind.Range
				if (
					self.mathtype is not spux.MathType.Complex
					and self.rows == 1
					and self.cols == 1
				):
					socket_info |= {
						'default_min': self.domain.inf,
						'default_max': self.domain.sup,
					}
					## TODO: Handle discontinuities / disjointness / open boundaries.

			msg = f'Tried to generate an ExprSocket from a SymSymbol "{self.name}", but its unit ({self.unit}) is not a valid unit of its physical type ({self.physical_type}) (SimSymbol={self})'
			raise NotImplementedError(msg)

		msg = f'Tried to generate an ExprSocket from a SymSymbol "{self.name}", but its size ({self.rows} by {self.cols}) is incompatible with ExprSocket (SimSymbol={self})'
		raise NotImplementedError(msg)

	####################
	# - Operations: Raw Update
	####################
	def update(self, **kwargs) -> typ.Self:
		"""Create a new `SimSymbol`, such that the given keyword arguments override the existing values."""
		if not kwargs:
			return self

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
			domain=get_attr('domain'),
		)

	####################
	# - Operations: Comparison
	####################
	def compare(self, other: typ.Self) -> typ.Self:
		"""Whether this SimSymbol can be considered equivalent to another, and thus universally usable in arbitrary mathematical operations together.

		In particular, two attributes are ignored:
		- **Name**: The particluar choices of name are not generally important.
		- **Unit**: The particulars of unit equivilancy are not generally important; only that the `PhysicalType` is equal, and thus that they are compatible.

		While not usable in all cases, this method ends up being very helpful for simplifying certain checks that would otherwise take up a lot of space.
		"""
		return (
			self.mathtype is other.mathtype
			and self.physical_type is other.physical_type
			and self.compare_size(other)
			and self.domain == other.domain
		)

	def compare_size(self, other: typ.Self) -> typ.Self:
		"""Compare the size of this `SimSymbol` with another."""
		return self.rows == other.rows and self.cols == other.cols

	def compare_addable(
		self, other: typ.Self, allow_differing_unit: bool = False
	) -> bool:
		"""Whether two `SimSymbol`s can be added."""
		common = (
			self.compare_size(other.output)
			and self.physical_type is other.physical_type
			and not (
				self.physical_type is spux.NonPhysical
				and self.unit is not None
				and self.unit != other.unit
			)
			and not (
				other.physical_type is spux.NonPhysical
				and other.unit is not None
				and self.unit != other.unit
			)
		)
		if not allow_differing_unit:
			return common and self.output.unit == other.output.unit
		return common

	def compare_multiplicable(self, other: typ.Self) -> bool:
		"""Whether two `SimSymbol`s can be multiplied."""
		return self.shape_len == 0 or self.compare_size(other)

	def compare_exponentiable(self, other: typ.Self) -> bool:
		"""Whether two `SimSymbol`s can be exponentiated.

		"Hadamard Power" is defined for any combination of scalar/vector/matrix operands, for any `MathType` combination.
		The only important thing to check is that the exponent cannot have a physical unit.

		Sometimes, people write equations with units in the exponent.
		This is a notational shorthand that only works in the context of an implicit, cancelling factor.
		We reject such things.

		See https://physics.stackexchange.com/questions/109995/exponential-or-logarithm-of-a-dimensionful-quantity
		"""
		return (
			other.physical_type is spux.PhysicalType.NonPhysical and other.unit is None
		)

	####################
	# - Operations: Copying Setters
	####################
	def set_constant(self, constant_value: spux.SympyType) -> typ.Self:
		"""Set the constant value of this `SimSymbol`, by setting it as the only value in a `sp.FiniteSet` domain.

		The `constant_value` will be conformed and stripped (with `self.conform()`) before being injected into the new `sp.FiniteSet` domain.

		Warnings:
			Keep in mind that domains do not encode units, for practical reasons related to the diverging ways in which various `sp.Set` subclasses interpret units.

			This isn't noticeable in normal constant-symbol workflows, where the constant is retrieved using `self.constant_value` (which adds `self.unit_factor`).
			However, **remember that retrieving the domain directly won't add the unit**.

			Ye been warned!
		"""
		if self.is_constant:
			return self.update(
				domain=sp.FiniteSet(self.conform(constant_value, strip_unit=True))
			)

		msg = 'Tried to set constant value of non-constant SimSymbol.'
		raise ValueError(msg)

	####################
	# - Operations: Conforming Mappers
	####################
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
		if (self.rows > 1 or self.cols > 1) and not isinstance(
			res, sp.MatrixBase | sp.MatrixSymbol
		):
			res = res * sp.ImmutableMatrix.ones(self.rows, self.cols)

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

	####################
	# - Creation
	####################
	@staticmethod
	def from_expr(
		sym_name: SimSymbolName,
		expr: spux.SympyExpr,
		unit_expr: spux.SympyExpr,
		is_constant: bool = False,
		optional: bool = False,
	) -> typ.Self | None:
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
		mathtype = spux.MathType.from_expr(expr, optional=optional)
		if mathtype is None:
			return None

		# PhysicalType as "NonPhysical"
		## -> 'unit' still applies - but we can't guarantee a PhysicalType will.
		## -> Therefore, this is what we gotta do.
		if spux.uses_units(unit_expr):
			simplified_unit_expr = sp.simplify(unit_expr)
			expr_physical_type = spux.PhysicalType.from_unit(
				simplified_unit_expr, optional=True
			)

			physical_type = (
				spux.PhysicalType.NonPhysical
				if expr_physical_type is None
				else expr_physical_type
			)
			unit = simplified_unit_expr
		else:
			physical_type = spux.PhysicalType.NonPhysical
			unit = None

		# Rows/Cols from Expr (if Matrix)
		rows, cols = (
			expr.shape if isinstance(expr, sp.MatrixBase | sp.MatrixSymbol) else (1, 1)
		)

		return SimSymbol(
			sym_name=sym_name,
			mathtype=mathtype,
			physical_type=physical_type,
			unit=unit,
			rows=rows,
			cols=cols,
			is_constant=is_constant,
			exclude_zero=expr.is_zero is not None and not expr.is_zero,
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
