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

"""Implements `SimSymbol`, a convenient representation of a symbolic variable suiteable for use when describing various mathematical and numerical interfaces."""

import functools
import random
import typing as typ

import jax
import jax.numpy as jnp
import jaxtyping as jtyp
import numpy as np
import pydantic as pyd
import sympy as sp
import sympy.stats as sps
from sympy.tensor.array.expressions import ArraySymbol

from blender_maxwell.utils import logger, serialize
from blender_maxwell.utils import sympy_extra as spux
from blender_maxwell.utils.frozendict import frozendict
from blender_maxwell.utils.lru_method import method_lru

from .name import SimSymbolName
from .utils import unicode_superscript

log = logger.get(__name__)

MT = spux.MathType


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

	# Name | Type
	sym_name: SimSymbolName
	mathtype: MT = MT.Real
	physical_type: spux.PhysicalType = spux.PhysicalType.NonPhysical

	# Units
	## -> 'None' indicates that no particular unit has yet been chosen.
	## -> When 'self.physical_type' is NonPhysical, _no unit_ can be chosen.
	unit: spux.Unit | None = None

	####################
	# - Dimensionality
	####################
	# Size
	## -> 1*1: "Scalar".
	## -> n*1: "Vector".
	## -> 1*n: "Covector".
	## -> n*m: "Matrix".
	## -> n*m*...: "Tensor".
	rows: int = 1
	cols: int = 1
	depths: tuple[int, ...] = ()

	@functools.cached_property
	def is_scalar(self) -> bool:
		"""Whether the symbol represents a scalar."""
		return self.rows == self.cols == 1 and self.depths == ()

	@functools.cached_property
	def is_vector(self) -> bool:
		"""Whether the symbol represents a (column) vector."""
		return self.rows > 1 and self.cols == 1 and self.depths == ()

	@functools.cached_property
	def is_covector(self) -> bool:
		"""Whether the symbol represents a covector."""
		return self.rows == 1 and self.cols > 1 and self.depths == ()

	@functools.cached_property
	def is_matrix(self) -> bool:
		"""Whether the symbol represents a matrix, which isn't better described as a scalar/vector/covector."""
		return self.rows > 1 and self.cols > 1 and self.depths == ()

	@functools.cached_property
	def is_ndim(self) -> bool:
		"""Whether the symbol represents an n-dimensional tensor, which isn't better described as a scalar/vector/covector/matrix."""
		return self.depths != ()

	####################
	# - Domain
	####################
	# Representation of Valid Symbolic Domain
	## -> Declares the valid set of values that may be given to this symbol.
	## -> By convention, units are not encoded in the domain sp.Set.
	## -> 'sp.Set's are extremely expressive and cool.
	domain: spux.BlessedSet | None = None

	@functools.cached_property
	def domain_mat(self) -> sp.Set | sp.matrices.MatrixSet:
		"""Get the domain as a `MatrixSet`, if the symbol represents a vector/covector/matrix, otherwise .

		Raises:
			ValueError: If the symbol is an arbitrary n-dimensional tensor.
		"""
		if self.is_scalar:
			return self.domain
		if self.is_vector or self.is_covector or self.is_matrix:
			return self.domain.bset_mat(self.rows, self.cols)

		msg = f"Can't determine set representation of arbitrary n-dimensional tensor (SimSymbol = {self})"
		raise ValueError(msg)

	####################
	# - Stochastic Variables
	####################
	# Stochastic Sample Space | PDF
	## -> When stoch_var is set, this variable should be considered stochastic.
	## -> When stochastic, the SimSymbol must be real | unitless | [0,1] dm.
	stoch_var: spux.SympyExpr | None = None
	stoch_seed: int = 0

	@functools.cached_property
	def stoch_key_jax(self) -> jax._src.prng.PRNGKeyArray:
		"""The key guaranteeing a deterministic random sample based on `self.stoch_seed`."""
		return jax.random.key(self.stock_seed)

	@functools.cached_property
	def sample_space(self) -> sp.Set | None:
		"""The space of valid inputs to the PDF."""
		return self.stoch_var.pspace.set

	@functools.cached_property
	def pdf(self) -> spux.SympyExpr | None:
		"""The expression of probability density function.

		When `self.sample_space` is `None`, then this too returns `None`, since the symbol is not stochastic (and therefore has no "measurable space").
		"""
		if self.stoch_var is not None:
			return self.stoch_var.pspace.pdf
		return None

	@functools.cached_property
	def cdf(self) -> spux.SympyExpr | None:
		"""The expression of cumulative distribution function."""
		if self.stoch_var is not None:
			return sps.cdf(self.stoch_var)
		return None

	@functools.cached_property
	def pdf_jax(self) -> typ.Callable[[jax.Array], jax.Array] | None:
		"""The stochastic PDF as a `jax`-registered function."""
		if self.stoch_var is not None:
			return sp.lambdify(self.stoch_var, self.pdf, 'jax')
		return None

	@method_lru()
	def sample_np(self, repeat: int = 1) -> typ.Callable[[jax.Array], jax.Array] | None:
		"""Sample the stochastic variable `repeat` times, using 'numpy'."""
		if self.stoch_var is not None:
			sample_shape = (*self.shape, repeat)
			return sps.sample(self.stoch_var, size=sample_shape, library='numpy')
		return None

	@method_lru()
	def sample_jax(
		self, repeat: int = 1
	) -> typ.Callable[[jax.Array], jax.Array] | None:
		"""Sample the stochastic variable `repeat` times, using 'jax'.

		For now, this cannot be used in `@jit`, since it merely converts the output of `sample_np`.
		In the long run, we'll need a way of directly sampling `sympy` stochastic variables with `jax` functions
		"""
		if self.stoch_var is not None:
			return jnp.array(self.sample_np(repeat))
		return None

	# TODO: 'Ecosystem'!
	## -- Ideally we'd expand sympy.stats.sample to support 'jax' directly.
	## -- 'None' dims in InfoFlow would mean either cont. or stochastic.
	## -- Stochastic dims would be realized w/integer n argument.
	## -- I suppose a dedicated 'stochastic symbol' node would be warranted.
	## -- With a nice dropdown for cool distributions and cool jazz.

	####################
	# - Validators: Stochastic Variable
	####################
	@pyd.model_validator(mode='after')
	def randomize_stochvar_key(self) -> typ.Self:
		"""Select a random integer value for the stochastic seed.

		Repeated calls to `self.sample_jax()`
		"""
		if self.stoch_var is not None:
			self.stoch_seed = random.randint(0, 2**32)
		return self

	@pyd.model_validator(mode='after')
	def set_stochvar_output_space_to_domain(self) -> typ.Self:
		"""When the symbol is stochastic, set `self.domain` to the range of the stochastic variable's probability density function.

		All PDFs are real, unitless values defined on the closed interval `[0,1]`.
		Therefore, the symbol itself must conform to these preconditions.
		"""
		if self.stoch_var is not None:
			if (
				self.domain is None
				and self.physical_type is spux.PhysicalType.NonPhysical
				and self.unit is None
			):
				object.__setattr__(self, 'domain', spux.BlessedSet(sp.Interval(0, 1)))
			else:
				msg = 'Stochastic variables must be unitless'
				raise ValueError(msg)
		return self

	####################
	# - Validators: Set Domain
	####################
	@pyd.model_validator(mode='after')
	def set_undefined_domain_from_mathtype(self) -> typ.Self:
		"""When the domain is not set, then set it using the symbolic set of the MathType."""
		if self.domain is None:
			object.__setattr__(
				self, 'domain', spux.BlessedSet(self.mathtype.symbolic_set)
			)
		return self

	####################
	# - Validators: Asserters
	####################
	@pyd.model_validator(mode='after')
	def assert_domain_in_mathtype_set(self) -> typ.Self:
		"""Verify that the domain is a (non-strict) subset of the MathType."""
		if self.domain.bset is sp.EmptySet or self.domain.bset.issubset(
			self.mathtype.symbolic_set
		):
			return self

		msg = f'Domain {self.domain} is not in the mathtype {self.mathtype}'
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
		if self.depths == ():
			return {
				(1, 1): spux.NumberSize1D.Scalar,
				(2, 1): spux.NumberSize1D.Vec2,
				(3, 1): spux.NumberSize1D.Vec3,
				(4, 1): spux.NumberSize1D.Vec4,
			}.get((self.rows, self.cols))
		return None

	@functools.cached_property
	def shape(self) -> tuple[int, ...]:
		"""Deterministic chosen shape of this `SimSymbol`.

		Derived from `self.rows` and `self.cols`.

		Is never `None`; instead, empty tuple `()` is used.
		"""
		if self.depths == ():
			match (self.rows, self.cols):
				case (1, 1):
					return ()
				case (_, 1):
					return (self.rows,)
				case (_, _):
					return (self.rows, self.cols)

		return (self.rows, self.cols, *self.depths)

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
			case MT.Integer:
				mathtype_kwargs |= {'integer': True}
			case MT.Rational:
				mathtype_kwargs |= {'rational': True}
			case MT.Real:
				mathtype_kwargs |= {'real': True}
			case MT.Complex:
				mathtype_kwargs |= {'complex': True}

		# Non-Zero Assumption
		if self.is_nonzero:
			mathtype_kwargs |= {'nonzero': True}

		# Positive/Negative Assumption
		if self.mathtype is not MT.Complex:
			has_pos = self.domain & sp.Interval.open(0, sp.oo) is not sp.EmptySet
			has_neg = self.domain & sp.Interval.open(0, sp.oo) is not sp.EmptySet
			if has_pos and not has_neg:
				mathtype_kwargs |= {'positive': True}
			if has_neg and not has_pos:
				mathtype_kwargs |= {'negative': True}

		# Scalar: Return Symbol
		if self.is_scalar:
			return sp.Symbol(self.sym_name.name, **mathtype_kwargs)

		# Vector|Matrix: Return Matrix of Symbols
		## -> MatrixSymbol doesn't support assumptions.
		## -> This little construction does.
		if not self.is_ndim:
			return sp.ImmutableMatrix(
				[
					[
						sp.Symbol(
							self.sym_name.name + f'_{row}{col}', **mathtype_kwargs
						)
						for col in range(self.cols)
					]
					for row in range(self.rows)
				]
			)

		# Arbitrary Tensor: Just Return Symbol
		## -> Maybe we'll do the other stuff later to keep assumptions.
		## -> Maybe we'll retire matrix-of-symbol entirely instead.
		return self.sp_symbol_matsym

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
		if self.depths == ():
			return sp.MatrixSymbol(self.sym_name.name, self.rows, self.cols)
		return ArraySymbol(self.sym_name.name, self.shape)

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
			if self.unit in self.physical_type.valid_units or (
				self.unit is None and self.physical_type is None
			):
				socket_info = {
					'output_name': self.sym_name,
					# Socket Interface
					'size': self.size,
					'mathtype': self.mathtype,
					'physical_type': self.physical_type,
					# Defaults: Units
					'default_unit': self.unit,
					'default_symbols': [],
					'exclude_zero': 0 not in self.domain,
					# Domain Info
					'abs_min': self.domain.inf,
					'abs_max': self.domain.sup,
					'abs_min_closed': self.domain.min_closed,
					'abs_max_closed': self.domain.max_closed,
				}

				# Complex Domain: Closure of Imaginary Axis
				if self.mathtype is MT.Complex:
					socket_info |= {
						'abs_min_closed_im': self.domain.min_closed_im,
						'abs_max_closed_im': self.domain.min_closed_im,
					}

				# FlowKind.Range: Min/Max
				if self.mathtype is not MT.Complex and self.shape_len == 0:
					socket_info |= {
						'default_min': self.domain.inf,
						'default_max': self.domain.sup,
					}

				return socket_info

			msg = f'Tried to generate an ExprSocket from a SymSymbol "{self.name}", but its unit ({self.unit}) is not a valid unit of its physical type ({self.physical_type}) (SimSymbol={self})'
			raise NotImplementedError(msg)

		msg = f'Tried to generate an ExprSocket from a SymSymbol "{self.name}", but its size ({self.rows} by {self.cols}) is incompatible with ExprSocket (SimSymbol={self})'
		raise NotImplementedError(msg)

	####################
	# - Operations: Raw Update
	####################
	@method_lru()
	def _update(self, kwargs: frozendict) -> typ.Self:
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

	def update(self, **kwargs) -> typ.Self:
		"""Create a new `SimSymbol`, such that the given keyword arguments override the existing values."""
		return self._update(frozendict(kwargs))

	@method_lru(maxsize=512)
	def scale_to_unit_system(self, unit_system: spux.UnitSystem | None) -> typ.Self:
		"""Compute the SimSymbol resulting from the unit system conversion."""
		if self.unit is not None:
			new_unit = spux.convert_to_unit_system(self.unit, unit_system)

			scaling_factor = spux.convert_to_unit_system(
				self.unit, unit_system, strip_units=True
			)
			return self.update(
				unit=new_unit,
				domain=self.domain * scaling_factor,
			)
		return self

	####################
	# - Operations: Comparison
	####################
	@method_lru(maxsize=256)
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

	@method_lru(maxsize=256)
	def compare_size(self, other: typ.Self) -> typ.Self:
		"""Compare the size of this `SimSymbol` with another."""
		return (
			self.rows == other.rows
			and self.cols == other.cols
			and self.depths == other.depths
		)

	@method_lru(maxsize=256)
	def compare_addable(
		self, other: typ.Self, allow_differing_unit: bool = False
	) -> bool:
		"""Whether two `SimSymbol`s can be added."""
		common = (
			self.compare_size(other)
			and self.physical_type is other.physical_type
			and not (
				self.physical_type is spux.PhysicalType.NonPhysical
				and self.unit is not None
				and self.unit != other.unit
			)
			and not (
				other.physical_type is spux.PhysicalType.NonPhysical
				and other.unit is not None
				and self.unit != other.unit
			)
		)
		if not allow_differing_unit:
			return common and self.unit == other.unit
		return common

	@method_lru(maxsize=256)
	def compare_multiplicable(self, other: typ.Self) -> bool:
		"""Whether two `SimSymbol`s can be multiplied."""
		return self.shape_len == 0 or other.shape_len == 0 or self.compare_size(other)

	@method_lru(maxsize=256)
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
	# - Operations: Conforming Mappers
	####################
	def conform(
		self, obj: np.ndarray | jax.Array | spux.SympyType, strip_unit: bool = False
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
		if isinstance(obj, np.ndarray | jax.Array):
			if obj.shape == ():  # noqa: SIM108
				res = sp.S(obj.item(0))
			else:
				res = sp.S(np.array(obj))
		else:
			res = sp.S(obj)

		# Unit Conversion
		match (spux.uses_units(res), self.unit is not None):
			case (True, True):
				res = spux.scale_to_unit(res, self.unit) * self.unit

			case (False, True):
				res = res * self.unit

			case (True, False):
				res = spux.strip_unit_system(res)

		if strip_unit:
			res = spux.strip_unit_system(res)

		# Broadcast Expansion
		if (
			self.depths == ()
			and (self.rows > 1 or self.cols > 1)
			and not isinstance(res, sp.MatrixBase | sp.MatrixSymbol)
		):
			res = sp.ImmutableMatrix.ones(self.rows, self.cols).applyfunc(
				lambda el: 5 * el
			)

		return res

	@method_lru(maxsize=256)
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
	@functools.lru_cache(maxsize=512)
	@staticmethod
	def from_expr(
		sym_name: SimSymbolName,
		expr: spux.SympyExpr,
		unit_expr: spux.SympyExpr,
		new_domain: spux.BlessedSet | None = None,
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
		mathtype = MT.from_expr(expr, optional=optional)
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

		# Set Domain
		domain = (
			new_domain
			if new_domain is not None
			else spux.BlessedSet(mathtype.symbolic_set)
		)

		return SimSymbol(
			sym_name=sym_name,
			mathtype=mathtype,
			physical_type=physical_type,
			unit=unit,
			rows=rows,
			cols=cols,
			domain=(
				domain - sp.FiniteSet(0)
				if expr.is_zero is not None and not expr.is_zero
				else domain
			),
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
