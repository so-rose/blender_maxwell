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

"""Implements a wrapper for the use of `sympy` sets in deducing the image of a key selection of functions/operations.

Using sets to represent valid domains of symbols delves into territory that `sympy` only theoretically supports.
The internal `sympy.sets.setexpr.SetExpr`, which drives most of the available simplifications, is only of help in some of the absolute most simple cases.
To remedy this, we've decided to "bless" a few sets that are absolutely essential for our needs.

In total, we're left with a distinctly usable object capable of tracking symbolic domains of validity through extensive mathematical operations.
"""

import functools
import itertools
import operator
import typing as typ
from fractions import Fraction

import jax
import jax.numpy as jnp
import pydantic as pyd
import sympy as sp
from sympy.sets.setexpr import SetExpr

from blender_maxwell.utils.lru_method import method_lru

from .. import logger
from .math_type import MathType as MT  # noqa: N817
from .sympy_expr import ScalarUnitlessComplexExpr, SympyExpr

log = logger.get(__name__)


####################
# - Types
####################
Scalar: typ.TypeAlias = ScalarUnitlessComplexExpr
MatrixSet: typ.TypeAlias = sp.matrices.MatrixSet
ComplexRegion: typ.TypeAlias = sp.sets.fancysets.CartesianComplexRegion

# Valid BlessedSet Types:
## Trivial:
## - sp.EmptySet: Null Set
## Points:
## - sp.FiniteSet  ## -> For some reason, no work, but it's blessed too.
## Region:
## - sp.Range (stride=1): [a,b]_Z w/1-spacing
## - sp.Interval
## - ComplexRegion: tuple[.args[0], .args[1]] w/blessed args.
## - sp.Complexes: Inf ComplexRegion unfortunately simplifies to Complexes.
## Composite:
## - sp.Union[*, *, ...]: Arbitrary depth.
## TODO: Bless ConditionSets (for arbitrarily expressive domain shapes).

BlessedSetType: typ.TypeAlias = SympyExpr  # SympyObj(
# instance_of={
# sp.EmptySet,
# sp.FiniteSet,
# sp.Range,
# sp.Interval,
# ComplexRegion,
# sp.Complexes,
# sp.Union,
# }
# )
BlessedDomainOp: typ.TypeAlias = typ.Literal[
	operator.add, operator.sub, operator.mul, operator.truediv, operator.pow
]

# The following set types are "blessable" (aka. parseable to a BlessedSet):
## Universal:
## - sp.Naturals: Range(1, oo)
## - sp.Naturals0: Range(0, oo)
## - sp.Integers: Range(-oo, oo)
## - sp.Rationals: Interval(-oo, oo)  ## TODO: Custom rationals interval?
## - sp.Reals: Interval(-oo, oo)
## Composite:
## - MatrixSet[*]: Any parseable set.


####################
# - Bless Arbitrary Sets
####################
@functools.lru_cache(maxsize=8192)
def simplify_blessed_set(s: BlessedSetType) -> BlessedSetType:
	"""Perform some principled simplifications on computed `BlessedSetType`s."""
	# log.critical(s)

	sset = s

	if isinstance(sset, sp.FiniteSet):
		sset -= {-sp.oo, sp.oo, -sp.zoo, sp.zoo}
		# sset = sp.FiniteSet(*[sp.nsimplify(fs_el, tolerance=10**-7) for fs_el in sset])

	elif isinstance(sset, sp.Range):
		if sset.start == 0 and sset.stop == 0:
			return sp.EmptySet

	elif isinstance(sset, sp.Interval):
		return sp.Interval(
			sset.start,
			sset.end,
			# sp.nsimplify(sset.start, tolerance=10**-7),
			# sp.nsimplify(sset.end, tolerance=10**-7),
			left_open=sset.left_open,
			right_open=sset.right_open,
		)
		if sset.start == 0 and sset.stop == 0:
			return sp.EmptySet

	elif (
		isinstance(sset, ComplexRegion)
		and isinstance(sset.b_interval, sp.FiniteSet)
		and sset.b_interval == sp.FiniteSet(0)
	):
		sset = s.interval_a

	elif isinstance(sset, sp.Union):
		sset = sp.Union(*(simplify_blessed_set(arg) for arg in s.args))

	return sset


@functools.lru_cache(maxsize=8192)
def bless_set(s: sp.Set) -> BlessedSetType:  # noqa: C901, PLR0911, PLR0912, PLR0915
	"""Attempt to conform an arbitrary input set to an equivalent `BlessedSetType`.

	This may not always be possible.
	Therefore, as a fallback, we detect the `MT` of the given set and return its corresponding symbolic set.
	"""
	# log.critical(s)

	if isinstance(s, set | frozenset):
		return sp.S(s)
	# Explicitly Parse
	## -> There are some sets we don't want to special-case later.
	if s is sp.Naturals:
		return sp.Range(1, sp.oo, 1)
	if s is sp.Naturals0:
		return sp.Range(0, sp.oo, 1)
	if s is sp.Integers:
		return sp.Range(-sp.oo, sp.oo, 1)
	if s is sp.Rationals or s is sp.Reals:
		return sp.Interval(-sp.oo, sp.oo)
	if s is sp.Complexes:
		return sp.Complexes
		## -> ComplexRegion(R*R) unfortunately simplifies to Complexes.
		## -> This means that we unfortunately must resort to use of a tuple.
		return (sp.Interval(-sp.oo, sp.oo), sp.Interval(-sp.oo, sp.oo))

	if isinstance(s, MatrixSet):
		return bless_set(s.set)

	# Explicitly Refuse
	## -> There are some sets we simply cannot transcribe.
	error = False
	if isinstance(s, sp.Range) and s.step != 1:
		log.error(
			'Range set %s w/range != 1 cannot be parsed; expanding to step size of 1',
			str(s),
		)
		return simplify_blessed_set(sp.Range(s.start, s.stop, 1))

	if isinstance(s, sp.Intersection):
		log.error('Intersection set %s cannot be parsed', str(s))
		error = True

	elif isinstance(s, sp.ProductSet):
		log.error('Product set %s cannot be parsed', str(s))
		raise NotImplementedError

	elif isinstance(s, sp.Complement):
		# Remove Segments from Range
		## -> We construct an equivalent union of disjoint ranges.
		## -> We search for start/end points by linear search w/'in' operator.
		## -> We determine removed points by simple set intersection.
		## -> This works with absolutely any sp.Set - even bs like ImageSet!
		if isinstance(s.args[0], sp.Range):
			rng = s.args[0]
			other = s.args[1]

			start = next(el for el in rng if el not in other)
			end = next(el for el in reversed(rng) if el not in other)
			removed = rng & other

			return bless_set(
				functools.reduce(
					lambda S, rng_i: S | rng_i,  # noqa: N803
					[
						sp.Range(s + 1, e, 1) if s != e else sp.EmptySet
						for s, e in itertools.pairwise(
							sp.FiniteSet(start - 1) | removed | sp.FiniteSet(end + 1)
						)
					],
				)
			)

		# Punch Finite "Holes" in Interval
		## -> Interval - Interval simplifies just fine.
		## -> BUT, Sympy gives up when trying to "punch holes" using integers.
		if isinstance(s.args[0], sp.Interval) and isinstance(
			s.args[1], sp.FiniteSet | sp.Range
		):
			ivl = s.args[0]
			other = s.args[1]

			removed = ivl & other
			holes = sp.S(set(removed))

			return functools.reduce(lambda s, hole: s - {hole}, holes, ivl)

		# Split ComplexRegion
		## -> Just split along each axis.
		if isinstance(s.args[0], ComplexRegion):
			cpx = s.args[0]
			other = s.args[1]

			removed = cpx & other
			return bless_set(
				ComplexRegion(
					bless_set(cpx.a_interval - removed.a_interval)
					* bless_set(cpx.b_interval - removed.b_interval),
				)
			)

		log.error('Complement set %s cannot be parsed', str(s))
		error = True

	elif isinstance(s, sp.SymmetricDifference):
		log.error('Symmetric Difference set %s cannot be parsed', str(s))
		error = True

	elif isinstance(s, sp.DisjointUnion):
		log.error('Disjoint Union set %s cannot be parsed', str(s))
		raise NotImplementedError

	elif s is sp.UniversalSet:
		log.error('Universal set %s cannot be parsed', str(s))
		raise NotImplementedError

	elif isinstance(s, sp.sets.fancysets.ImageSet):
		log.error('Image set %s cannot be parsed', str(s))
		raise NotImplementedError

	elif isinstance(
		s, sp.sets.fancysets.PolarComplexRegion | sp.sets.powerset.PowerSet
	):
		log.error('Polar complex region set %s cannot be parsed', str(s))
		error = True

	elif isinstance(s, sp.sets.conditionset.ConditionSet):
		log.error('Condition set %s cannot be parsed', str(s))
		error = True

	if error:
		return bless_set(MT.from_symbolic_set(s).symbolic_set)
	return simplify_blessed_set(s)


####################
# - SetExpr Operations
####################
@functools.lru_cache(maxsize=8192)
def set_expr_op(
	op: BlessedDomainOp, lhs: BlessedSetType, rhs: BlessedSetType
) -> BlessedSetType:
	"""Compute the elementwise  between two sets using `SetExpr`.

	It is the user's responsibility to ensure the particular operation between `lhs` and `rhs` can actually be evaluated by `SetExpr`.
	While the constraint of `BlessedDomainOp` is a prerequisite, there is absolutely no guarantee that `SetExpr` can evaluate.

	Raises:
		TypeError: If the SetExpr did not simplify away the internal `ImageSet`, causing the computed set to not to be usable as a `BlessedSetType`.
	"""
	computed_set = simplify_blessed_set(op(SetExpr(lhs), SetExpr(rhs)).set)
	if isinstance(computed_set, sp.ImageSet):
		msg = 'SetExpr did not evaluate for {op} between {lhs} and {rhs}'
		raise TypeError(msg)

	return computed_set


####################
# - Blessed Set
####################
## -> Generated using 'sp.nsolve(sp.diff(sp.sinc(x)), x, -5.0)'
SINC_MINIMUM = -4.49340945790906


class BlessedSet(pyd.BaseModel):
	"""A wrapper enabling reliable elementwise operations on and between a constrained set of `sp.Set`s.

	In particular, it provides correct implementations of:

	- Enriched Set Operations: The usual `|` and `&`, along with a few extra well-defined semantics.
	- Unary Operations: Compute the set resulting from the application of a particular, single-argument function.
	- Binary Operations w/Scalar: Compute the set resulting from the application of an operation between all elements of a set, and a given scalar.
	- Minkowski Operations: Compute the set resulting from the application of an operation between all elements of two sets.
	"""

	model_config = pyd.ConfigDict(frozen=True)

	bset: BlessedSetType

	####################
	# - Creation
	####################
	def __init__(self, bset: typ.Self | sp.Set | set) -> None:
		if isinstance(bset, BlessedSet):
			super().__init__(bset=bset.bset)
		elif isinstance(bset, set | frozenset):
			super().__init__(bset=bless_set(frozenset(bset)))
		super().__init__(bset=bless_set(bset))

	@functools.lru_cache(maxsize=8192)
	@staticmethod
	def reals_to_complex(a: typ.Self | sp.Set, b: typ.Self | sp.Set) -> None:
		# log.critical([a, b])
		BS = BlessedSet
		return BS(ComplexRegion(BS(a).bset * BS(b).bset))

	####################
	# - Sympy
	####################
	def _sympy_(self) -> typ.Self:
		return self.bset

	####################
	# - Properties
	####################
	@functools.cached_property
	def real(self) -> typ.Self:
		"""The real subset of the represented set."""
		if isinstance(self.bset, ComplexRegion):
			return BlessedSet(self.bset.a_interval)
		return self

	@functools.cached_property
	def imag(self) -> typ.Self:
		"""The imaginary subset of the represented set."""
		if isinstance(self.bset, ComplexRegion):
			return BlessedSet(self.bset.b_interval)
		return BlessedSet(sp.FiniteSet(0))

	@functools.cached_property
	def is_empty(self) -> typ.Self:
		"""Whether this is the null set."""
		return self.bset is sp.EmptySet

	@functools.cached_property
	def is_nonzero(self) -> typ.Self:
		"""Whether `0` occurs in this set."""
		return 0 in self.bset

	@functools.cached_property
	def inf(self) -> typ.Self:
		"""The largest lower bound on this set.

		For complex sets, we define this across both real and imaginary axes.
		"""
		if self.bset is sp.EmptySet:
			msg = 'Empty set has no infimum'
			raise TypeError(msg)

		if isinstance(self.bset, sp.FiniteSet | sp.Range | sp.Interval):
			return self.bset.inf

		if isinstance(self.bset, ComplexRegion):
			return self.real.inf + sp.I * self.real.inf

		if isinstance(self.bset, sp.Complexes):
			return -sp.zoo

		if isinstance(self.bset, sp.Union):
			return min([BlessedSet(subset).inf for subset in self.bset.args])

		raise TypeError

	@functools.cached_property
	def sup(self) -> typ.Self:
		"""The smallest upper bound on this set.

		For complex sets, we define this across both real and imaginary axes.
		"""
		if self.bset is sp.EmptySet:
			msg = 'Empty set has no infimum'
			raise TypeError(msg)

		if isinstance(self.bset, sp.FiniteSet | sp.Range | sp.Interval):
			return self.bset.sup

		if isinstance(self.bset, ComplexRegion):
			return self.real.sup + sp.I * self.imag.sup

		if isinstance(self.bset, sp.Complexes):
			return sp.zoo

		if isinstance(self.bset, sp.Union):
			return min([BlessedSet(subset).inf for subset in self.bset.args])

		raise TypeError

	@functools.cached_property
	def min_closed(self) -> typ.Self:
		"""The closure of the largest lower bound.

		For complex sets, this refers to the closure of the real part.
		"""
		if self.bset is sp.EmptySet or self.bset is sp.Complexes:
			return False

		if isinstance(self.bset, sp.FiniteSet | sp.Range):
			return True

		if isinstance(self.bset, sp.Interval):
			return not self.bset.left_open

		if isinstance(self.bset, ComplexRegion):
			return self.real.min_closed

		if isinstance(self.bset, sp.Union):
			real_subsets = sorted(
				[BlessedSet(subset).real for subset in self.bset.args],
				lambda els: els.inf,
			)
			return real_subsets[0].inf in real_subsets[0]

		raise TypeError

	@functools.cached_property
	def max_closed(self) -> typ.Self:
		"""The closure of the smallest upper bound.

		For complex sets, this refers to the closure of the real part.
		"""
		if self.bset is sp.EmptySet or self.bset is sp.Complexes:
			return False

		if isinstance(self.bset, sp.FiniteSet | sp.Range):
			return True

		if isinstance(self.bset, sp.Interval):
			return not self.bset.right_open

		if isinstance(self.bset, ComplexRegion):
			return self.real.max_closed

		if isinstance(self.bset, sp.Union):
			real_subsets = sorted(
				[BlessedSet(subset).real for subset in self.bset.args],
				lambda els: els.sup,
			)
			return real_subsets[0].sup in real_subsets[0]

		raise TypeError

	@functools.cached_property
	def min_closed_im(self) -> bool:
		"""The closure of the largest lower bound of the imaginary axis.

		Purely real sets will generally have closed imaginary axes, as it is just the point `0`.
		"""
		if self.bset is sp.EmptySet or self.bset is sp.Complexes:
			return False

		if isinstance(self.bset, sp.FiniteSet | sp.Range | sp.Interval):
			return True  ## {0} is closed

		if isinstance(self.bset, ComplexRegion):
			return self.imag.min_closed

		if isinstance(self.bset, sp.Union):
			imag_subsets = sorted(
				[BlessedSet(subset).imag for subset in self.bset.args],
				lambda els: els.inf,
			)
			return imag_subsets[0].inf in imag_subsets[0]

		raise TypeError

	@functools.cached_property
	def max_closed_im(self) -> bool:
		"""The closure of the smallest upper bound of the imaginary axis.

		Purely real sets will generally have closed imaginary axes, as it is just the point `0`.
		"""
		if self.bset is sp.EmptySet or self.bset is sp.Complexes:
			return False

		if isinstance(self.bset, sp.FiniteSet | sp.Range | sp.Interval):
			return True  ## {0} is closed

		if isinstance(self.bset, ComplexRegion):
			return self.imag.max_closed

		if isinstance(self.bset, sp.Union):
			imag_subsets = sorted(
				[BlessedSet(subset).imag for subset in self.bset.args],
				lambda els: els.inf,
			)
			return imag_subsets[0].sup in imag_subsets[0]

		raise TypeError

	####################
	# - Methods
	####################
	@method_lru()
	def bset_mat(self, rows: int, cols: int) -> typ.Self:
		# log.critical([self, rows, cols])
		if rows == cols == 1:
			return self.bset
		return MatrixSet(rows, cols, BlessedSet(self.bset))

	def sample_uniform_jax(self, key, sample_shape: tuple[int, ...]) -> jax.Array:
		"""Sample elements uniformly from this set, returning a `jax.Array` of the specified `sample_shape`.

		- `FiniteSet`: Select from individual elements.
		- `Range`: Select from integers within the range.
		- `Interval`: Sample from the interior.
		- `ComplexRegion`: Sample real/imag axes seperately and combine to a complex array.
		- `Union` of `Interval`: Sample each disjoint `Interval` seperately, then use their relative size to weigh which individually sampled `Interval` to select an element from.
		"""
		if self.bset is sp.EmptySet:
			msg = 'Cant sample an empty set'
			raise TypeError(msg)

		if isinstance(self.bset, sp.FiniteSet):
			if (
				-sp.oo in self.bset
				or sp.oo in self.bset
				or -sp.zoo in self.bset
				or sp.zoo in self.bset
			):
				msg = 'Cant uniformly sample a finite set with infinite elements.'
				raise ValueError(msg)

			mathtype = MT.combine(self.bset)
			finite_domain = jnp.array(list(self.bset), dtype=mathtype.dtype)

			return jax.random.choice(key, sample_shape, a=finite_domain)

		if isinstance(self.bset, sp.Range):
			if self.bset.start == -sp.oo or self.bset.stop == sp.oo:
				msg = 'Cant uniformly sample a range with infinite bounds.'
				raise ValueError(msg)

			start = int(self.bset.start)
			stop_excl = int(self.bset.stop)

			return jax.random.randint(key, sample_shape, minval=start, maxval=stop_excl)

		if isinstance(self.bset, sp.Interval):
			if self.bset.start == -sp.oo or self.bset.end == sp.oo:
				msg = 'Cant uniformly sample an infinite Interval.'
				raise ValueError(msg)

			start = float(self.bset.start)
			end = float(self.bset.end)

			return jax.random.uniform(
				key,
				sample_shape,
				minval=start,
				maxval=end,
			)

		if isinstance(self.bset, ComplexRegion):
			re = BlessedSet(self.bset.a_interval).sample_uniform_jax(sample_shape)
			im = BlessedSet(self.bset.b_interval).sample_uniform_jax(sample_shape)

			return re + 1j * im

		if isinstance(self.bset, sp.Complexes):
			msg = 'Cant uniformly sample a set with infinite elements.'
			raise TypeError(msg)

		if isinstance(self.bset, sp.Union):
			if all(
				isinstance(disjoint_subset, sp.Interval)
				for disjoint_subset in self.bset.args
			):
				# Deduce Subkeys
				## -> This enables deterministic sampling of subsets.
				subkeys = jax.random.split(key, len(self.bset.args) + 1)

				# Deduce Relative Weights
				## -> This enables proper weighting of subset samples.
				weights = jnp.array(
					[
						disjoint_subset.sup - disjoint_subset.inf
						for disjoint_subset in self.bset.args
					]
				)
				weights /= sum(weights)

				# Assemble Uniform Samples of Disjoint Subsets
				## -> Disjointness allows independent sampling.
				disjoint_subset_samples = [
					BlessedSet(disjoint_subset).sample_uniform_jax(
						subkeys[i], sample_shape
					)
					for i, disjoint_subset in enumerate(self.bset.args)
				]

				# Perform Weighted Selection of Disjoint Uniform Samplings
				## -> We randomly generate a weighted index array.
				## -> This produces an index per-sample_shape pos.
				## -> This per-pos index selects values from disjoint samplings.
				## -> Thus, we've achieved weighted uniform disjoint sampling.
				weighted_idx_sampling = jax.random.choice(
					subkeys[-1],
					shape=sample_shape,
					a=jnp.arange(0, len(self.bset.args)),
					p=weights,
				)
				return jnp.choose(
					weighted_idx_sampling, choices=disjoint_subset_samples
				)

			raise NotImplementedError

		raise TypeError

	####################
	# - Set Operations
	####################
	def __contains__(self, value: typ.Self | sp.MatrixBase | typ.Any) -> typ.Self:
		"""Deduce whether `value` is contained within this BlessedSet.

		Generally, `in` only refers to "membership".
		However, we provide two special conveniences to make it easier to work with `BlessedSet`s in practice.

		- Breaking from the pure mathematical interpretation of membership, `value` is a `BlessedSet`, then it is considered to be "contained" within `self` it it is either the `sp.EmptySet` **or a subset** of `self.bset`.
		- If `value` is itself a `MatrixBase`, then we will wrap `self.bset` in an appropriately shaped `MatrixSet` before determining membership.
		"""
		# log.critical([self, value])
		# BlessedSet: Compute Subset
		## -> In mathematics, 'has me as a subset' is an explicit op.
		## -> Generally, there's a difference btwn "element of" and "subset".
		## -> BUT, here, it can be well-defined of a 'BlessedSet' wrapper.
		## -> Therefore, we can match Python's ex. string behavior.
		## -> This convenience eliminates a ton of boilerplate.
		## -> NOTE: The EmptySet check is also easy to miss.
		if isinstance(value, BlessedSet):
			return value.bset is sp.EmptySet or value.bset.issubset(self.bset)

		# MatrixSet: Deduce Shape-Uniform Membership
		## -> In our system, we've decided that matrices must have uniform dms.
		## -> This prevents a lot of very elaborate matrix domain math.
		## -> This also lets us express matrix domain math as normal domain math.
		## -> However, we still need to work with matrix domains _as matrices_.
		## -> Aka. 'sp.Matrix([1, 2, 3]) in BlessedSet(sp.Reals)' must work.
		## -> This avoids a ton of fragile boilerplate everywhere.
		if isinstance(value, sp.MatrixBase):
			return value in MatrixSet(*value.shape, self.bset)

		return value in self.bset

	def __or__(self, other: typ.Self | sp.Set | set) -> typ.Self:
		"""Compute the set union."""
		# log.critical([self, other])
		if isinstance(other, BlessedSet):
			return BlessedSet(self.bset | other.bset)
		return BlessedSet(self.bset | other)

	def __and__(self, other: typ.Self | sp.Set | set) -> typ.Self:
		"""Compute the set intersection."""
		# log.critical([self, other])
		if isinstance(other, BlessedSet):
			return BlessedSet(self.bset & other.bset)
		return BlessedSet(self.bset & other)

	####################
	# - Unary Operations
	####################
	def __abs__(self) -> typ.Self:
		return self.abs

	@functools.cached_property
	def abs(self) -> typ.Self:
		"""Apply the absolute value to all set elements."""
		# log.critical(self)

		s = self.bset
		nset = s

		# Trivial
		if s is sp.EmptySet:
			nset = sp.EmptySet

		# Points
		elif isinstance(s, sp.FiniteSet):
			nset = sp.FiniteSet(*[abs(el) for el in self.bset])

		# Ranges
		elif isinstance(s, sp.Range):
			start, _, end = sorted([sp.S(0), abs(s.inf), abs(s.sup)])
			if start == end:  # noqa: SIM108
				nset = sp.FiniteSet(start)
			else:
				nset = sp.Range(start, end + 1)

		elif isinstance(s, sp.Interval):
			(start, start_open), (end, end_open) = sorted(
				[
					(abs(s.start), s.left_open),
					(abs(s.end), s.right_open),
					## Can't use the same trick, as open/closed bounds don't sort.
				],
				key=lambda el: el[0],
			)

			if 0 in s:
				nset = sp.Interval(0, end, left_open=False, right_open=end_open)
			else:
				nset = sp.Interval(
					start,
					end,
					left_open=start_open,
					right_open=end_open,
				)

		elif isinstance(s, ComplexRegion):
			nset = ((self.real**2 + self.imag**2) ** sp.Rational(1, 2)).bset

		elif s is sp.Complexes:
			nset = sp.Interval(0, sp.oo)

		# Union: Recurse
		elif isinstance(s, sp.Union):
			nset = sp.Union(*[abs(arg).bset for arg in s.args])

		else:
			msg = f'abs() not implemented for set {s}'
			raise TypeError(msg)

		return BlessedSet(nset)

	@functools.cached_property
	def reciprocal(self) -> typ.Self:
		"""Compute the `BlessedSet` resulting from applying the reciprocal function $1/x$ to each element."""
		s = self.bset
		nset = s

		# Trivial
		## -> Sympy considers zero-divison as sp.S(x) / 0 = sp.zoo.
		## -> This 'complex infinity' is just sign-matching complex reals.
		## -> BUT, we need to do numerical stuff, and 'inf' is a problem.
		## -> Therefore, RESET EVERYTHING to EmptySet if */0 can happen.
		if s is sp.EmptySet or 0 in s:
			nset = sp.EmptySet

		# Recursive
		elif isinstance(s, sp.Union):
			nset = sp.Union(*[BlessedSet(arg).reciprocal.bset for arg in s.args])

		# Points
		elif isinstance(s, sp.FiniteSet):
			nset = sp.FiniteSet(*[1 / el for el in s])

		# Ranges
		elif isinstance(s, sp.Range):
			start, end = sorted([1 / s.inf, 1 / s.sup])
			if start == end:
				nset = sp.FiniteSet(start)
			elif start.is_integer and end.is_integer:
				nset = sp.Range(start, end + 1, 1)
			else:
				nset = sp.Interval(start, end, 1)

		elif isinstance(s, sp.Interval):
			(start, start_open), (end, end_open) = sorted(
				[
					(1 / s.start, s.left_open),
					(1 / s.end, s.right_open),
				],
				key=lambda el: el[0],
			)
			nset = sp.Interval(
				start,
				end,
				left_open=start_open,
				right_open=end_open,
			)

		elif isinstance(s, ComplexRegion):
			denominator = self.real**2 + self.imag**2
			nset = BlessedSet.reals_to_complex(
				self.real / denominator,
				-self.imag / denominator,
			).bset

		elif s is sp.Complexes:
			nset = sp.Reals

		else:
			msg = f'abs() not implemented for set {s}'
			raise ValueError(msg)

		# log.critical(BlessedSet(nset - {0}))
		return BlessedSet(nset - {0})

	@functools.cached_property
	def cos(self) -> typ.Self:
		r"""Compute the `BlessedSet` resulting from applying the reciprocal function $\cos x$ to each element."""
		# log.critical(self)

		s = self.bset
		# Trivial
		if s is sp.EmptySet:
			nset = sp.EmptySet

		# Points
		elif isinstance(s, sp.FiniteSet):
			nset = sp.FiniteSet(*[sp.cos(el) for el in s])

		# Ranges
		elif isinstance(s, sp.Range):
			x = sp.Symbol('x', real=True)
			nset = sp.calculus.util.function_range(
				sp.cos(x), x, sp.Interval(s.inf, s.sup)
			)
			## TODO: Don't just cast to Interval.
			## -- Easily infinite points.
			## -- Would need a blessed ConditionSet.

		elif isinstance(s, sp.Interval):
			x = sp.Symbol('x', real=True)
			nset = sp.calculus.util.function_range(sp.cos(x), x, s)

		elif isinstance(s, ComplexRegion):
			## TODO: cos(x)*cosh(y) - sin(x)*sinh(y)
			log.error(
				'cos(x) not (yet) implemented for complex region: %s. Falling back to sp.Complexes.',
				str(s),
			)
			nset = sp.Complexes

		elif s is sp.Complexes:
			## -> Applying cos(x)*cosh(y) - sin(x)*sinh(y) gives all of C.
			nset = sp.Complexes

		# Unions: Recurse
		elif isinstance(s, sp.Union):
			nset = sp.Union(*(BlessedSet(arg).cos.bset for arg in s.args))

		else:
			msg = f'cos()/sin() not implemented for set {s}'
			raise TypeError(msg)

		# log.critical(BlessedSet(nset))
		return BlessedSet(nset)

	@functools.cached_property
	def sin(self) -> typ.Self:
		r"""Compute the `BlessedSet` resulting from applying the reciprocal function $\cos x$ to each element."""
		# log.critical(self)

		return (self - sp.pi / 2).cos

	@functools.cached_property
	def arctan(self) -> typ.Self:
		r"""Compute the `BlessedSet` resulting from applying the reciprocal function $\cos x$ to each element."""
		# log.critical(self)

		s = self.bset

		# Trivial
		if s is sp.EmptySet:
			nset = sp.EmptySet

		# Points
		elif isinstance(s, sp.FiniteSet):
			nset = sp.FiniteSet(*[sp.arctan(el) for el in s])

		elif isinstance(s, sp.Range):
			x = sp.Symbol('x', real=True)
			nset = sp.calculus.util.function_range(
				sp.arctan(x), x, sp.Interval(s.inf, s.sup)
			)
			## TODO: It's more a bunch of points than really continuous...
			## -- FiniteSet might get way too big, though.

		elif isinstance(s, sp.Interval):
			x = sp.Symbol('x', real=True)
			nset = sp.calculus.util.function_range(sp.arctan(x), x, s)

		elif isinstance(s, ComplexRegion) or s is sp.Complexes:
			raise NotImplementedError

		# Unions: Recurse
		elif isinstance(s, sp.Union):
			nset = sp.Union(*(BlessedSet(arg).arctan for arg in s.args))

		else:
			msg = f'arctan() not implemented for set {s}'
			raise ValueError(msg)

		return BlessedSet(nset)

	@functools.cached_property
	def sinc(self) -> typ.Self:
		# log.critical(self)

		s = self.bset
		if isinstance(s, ComplexRegion) or s is sp.Complexes:
			## TODO: Handle better
			return sp.Complexes

		return BlessedSet(sp.Interval(SINC_MINIMUM, 1))

	@functools.cached_property
	def arg(self) -> typ.Self:
		"""Compute the set resulting from applying the complex argument to all elements."""
		# log.critical(self)

		s = self.bset

		# Trivial
		## -> arg of 0 is undefined just like zero-division.
		if s is sp.EmptySet or 0 in s:
			nset = sp.EmptySet

		# Points
		if isinstance(s, sp.FiniteSet):
			nset = sp.FiniteSet(*[sp.arg(el) for el in s])

		elif isinstance(s, sp.Range | sp.Interval):
			if s.inf > 0 and s.sup > 0:
				nset = sp.FiniteSet(0)
			if s.inf < 0 and s.sup < 0:
				nset = sp.FiniteSet(sp.pi)
			if s.inf < 0 and s.sup > 0:
				nset = sp.FiniteSet(0, sp.pi)
			raise TypeError

		elif isinstance(s, ComplexRegion):
			_q1_q4 = ComplexRegion(sp.Interval.open(0, sp.oo) * sp.Reals)
			_q2 = ComplexRegion(sp.Interval.open(-sp.oo, 0) * sp.Interval(0, sp.oo))
			_q3 = ComplexRegion(
				sp.Interval.open(-sp.oo, 0) * sp.Interval.open(0, sp.oo)
			)
			_pos_im_axis = ComplexRegion(sp.FiniteSet(0) * sp.Interval.open(0, sp.oo))
			_neg_im_axis = ComplexRegion(sp.FiniteSet(0) * sp.Interval.open(-sp.oo, 0))

			q1_q4 = s & _q1_q4
			q2 = s & _q2
			q3 = s & _q3
			pos_im_axis = s & _pos_im_axis
			neg_im_axis = s & _neg_im_axis

			nset = (
				(q1_q4.imag / q1_q4.real).arctan
				| (q2.imag / q2.real + sp.pi).arctan
				| (q3.imag / q3.real - sp.pi).arctan
				| (
					sp.FiniteSet(sp.EmptySet)
					if pos_im_axis.is_empty
					else sp.FiniteSet(sp.pi / 2)
				)
				| (
					sp.FiniteSet(sp.EmptySet)
					if neg_im_axis.is_empty
					else sp.FiniteSet(-sp.pi / 2)
				)
				## origin is unimportant here
			)

		elif s is sp.Complexes:
			nset = bless_set(sp.Reals)

		# Unions: Recurse
		elif isinstance(s, sp.Union):
			nset = sp.Union(*(BlessedSet(arg).arg.bset for arg in s.args))

		else:
			msg = f'arg() not implemented for set {s}'
			raise TypeError(msg)

		# log.critical(BlessedSet(nset))
		return BlessedSet(nset)

	####################
	# - Operation w/Scalar
	####################
	@method_lru()
	def _operate_scalar(self, op: BlessedDomainOp, scalar: Scalar) -> typ.Self:  # noqa: C901, PLR0915, PLR0912
		"""Compute the set resulting from applying an operation by a scalar to each set element."""
		log.critical(['SCALAR', self.bset, op, scalar])

		s = self.bset
		# Trivial
		if s is sp.EmptySet:
			nset = sp.EmptySet

		# Operator-Specific
		## -> This is both an optimization, and to protect SetExpr.
		elif op in [operator.add, operator.sub] and scalar == 0:
			nset = s

		elif op is operator.mul and scalar == 0:
			nset = sp.FiniteSet(0)

		elif op is operator.mul and scalar == 1:
			nset = s

		elif op is operator.truediv and scalar == 0:
			nset = sp.EmptySet

		elif op is operator.truediv and scalar == 1:
			nset = s

		elif op is operator.pow and scalar == 0:
			nset = sp.FiniteSet(1)

		elif op is operator.pow and scalar == 1:
			nset = s

		# Points
		## -> SetExpr works just fine here.
		elif isinstance(s, sp.FiniteSet):
			nset = set_expr_op(op, s, sp.FiniteSet(scalar))

		# Range
		## -> SetExpr gives up quite thoroughly whenever Range is involved.
		elif isinstance(s, sp.Range):
			start, stop = sorted([op(s.start, scalar), op(s.stop, scalar)])
			if start == stop:  # noqa: SIM108
				nset = sp.FiniteSet(start)
			else:
				nset = sp.Range(start, stop, 1)

		# Points
		## -> SetExpr w/Reals is just fine.
		## -> SetExpr w/Complexes is NOT fine.
		elif isinstance(s, sp.Interval):
			if sp.im(scalar) == 0:
				nset = set_expr_op(op, s, sp.FiniteSet(scalar))
			else:
				A = BlessedSet(s)
				c = sp.re(scalar)
				d = sp.im(scalar)

				if op in [operator.add, operator.sub]:
					nset = BlessedSet.reals_to_complex(
						self._operate_scalar(s, c),
						sp.FiniteSet(op(0, d)),
					).bset

				elif op is operator.mul:
					nset = BlessedSet.reals_to_complex(
						A * c,
						A * d,
					).bset

				elif op is operator.truediv:
					denominator = c**2 + d**2
					nset = BlessedSet.reals_to_complex(
						(A * c) / denominator,
						(A * d) / denominator,
					).bset

				elif op is operator.pow:
					log.error(
						'Exponentiation of Intervals by complex numbers is not (yet) supported; falling back to the entire set of complex numbers'
					)
					nset = sp.Complexes

		elif isinstance(s, ComplexRegion) or s is sp.Complexes:
			# Extract Complex Elements
			if s is sp.Complexes:
				A = BlessedSet(sp.Reals)
				B = BlessedSet(sp.Reals)
			else:
				A = self.real
				B = self.imag

			if isinstance(scalar, complex):
				c = sp.re(scalar)
				d = sp.im(scalar)
			else:
				c = scalar
				d = 0

			# + | -: Seperable
			if op in [operator.add, operator.sub]:
				nset = BlessedSet.reals_to_complex(
					A._operate_scalar(op, scalar),  # noqa: SLF001
					B._operate_scalar(op, scalar),  # noqa: SLF001
				).bset

			# *: Standard Arithmetic Rules
			elif op is operator.mul:
				nset = BlessedSet.reals_to_complex(
					A * c - B * d,
					B * c + A * d,
				).bset

			# /: Standard Arithmetic Rules
			elif op is operator.truediv:
				denominator = c**2 + d**2
				nset = BlessedSet.reals_to_complex(
					(A * c + B * d) / denominator,
					(B * c + A * d) / denominator,
				).bset

			# **: Complex Exponentiation
			## -> In the generic sense, this is a hell of a function.
			if op is operator.pow:
				# Trivial Cases
				if scalar == 0:
					_nset = sp.FiniteSet(1)

				elif scalar == 1:
					_nset = s

				# Extract Absolute Value of Exponent
				## -> Later, sign decides whether reciprocal will be applied.
				## -> For now, the abs() is what we need to make decisions.
				abs_scalar = abs(scalar)
				sgn_scalar = 1 if scalar >= 0 else -1

				# Complex | Integer
				## -> Apply De Moivre's Formula
				if abs_scalar.is_integer:
					N = int(abs_scalar)
					r_N = abs(self) ** N
					arg_N = N * self.arg

					_nset = BlessedSet.reals_to_complex(
						r_N * arg_N.cos,
						r_N * arg_N.sin,
					)

				# Complex | Rational
				elif isinstance(abs_scalar, Fraction):
					log.error(
						'Exponentiation of set w/rational numbers is not (yet) supported; falling back to the entire set of complex numbers'
					)
					_nset = sp.Complexes

				# Complex | Real
				if abs_scalar.is_rational:
					log.error(
						'Exponentiation of set w/real numbers is not (yet) supported; falling back to the entire set of complex numbers'
					)
					_nset = sp.Complexes

				# Complex | Complex
				elif d != 0:
					log.error(
						'Exponentiation of set w/complex number is not (yet) supported; falling back to the entire set of complex numbers'
					)
					_nset = sp.Complexes

				# Deduce Reciprocal (if exponent is negative)
				if sgn_scalar == 1:
					nset = _nset
				nset = BlessedSet(_nset).reciprocal.bset

		# Unions: Recurse
		elif isinstance(s, sp.Union):
			nset = sp.Union(
				*(BlessedSet(arg)._operate_scalar(op, scalar) for arg in s.args)  # noqa: SLF001
			)
		else:
			raise NotImplementedError

		log.critical(['SCALAR DONE', nset])
		return BlessedSet(nset)

	####################
	# - Operation w/Other Sets
	####################
	@method_lru()
	def _operate_minkowski(self, op: BlessedDomainOp, _rhs: typ.Self) -> typ.Self:  # noqa: PLR0915, C901, PLR0912
		"""Compute the set resulting from applying an operation by a scalar to each set element."""
		log.critical(['MINKOWSKI', self.bset, op, _rhs])

		lhs = self.bset
		if isinstance(_rhs, sp.Set | set):  # noqa: SIM108
			rhs = BlessedSet(_rhs).bset
		else:
			rhs = _rhs.bset

		# Trivial
		if lhs is sp.EmptySet:
			nset = rhs
		elif rhs is sp.EmptySet:
			nset = lhs

		elif op is operator.truediv and not rhs.is_nonzero:
			nset = sp.EmptySet

		# Unions: Recurse
		## -> For narrowing reasons, we do this before other checks.
		elif isinstance(lhs, sp.Union) and isinstance(rhs, sp.Union):
			nset = sp.Union(
				*[
					BlessedSet(l_arg)._operate_minkowski(op, r_arg)  # noqa: SLF001
					for l_arg, r_arg in itertools.product(lhs.args, rhs.args)
				]
			)
		elif isinstance(lhs, sp.Union):
			nset = sp.Union(
				*[BlessedSet(l_arg)._operate_minkowski(op, rhs) for l_arg in lhs.args]  # noqa: SLF001
			)
		elif isinstance(rhs, sp.Union):
			nset = sp.Union(
				*[BlessedSet(lhs)._operate_minkowski(op, r_arg) for r_arg in rhs.args]  # noqa: SLF001
			)

		# Complex: Recurse
		## -> We've eliminated EmptySet, Union.
		## -> We're left to account for Complex*, FiniteSet, Range, Interval.
		elif any(isinstance(s, ComplexRegion) or s is sp.Complexes for s in [lhs, rhs]):
			if isinstance(lhs, ComplexRegion) or lhs is sp.Complexes:
				A = BlessedSet(lhs.a_interval)
				B = BlessedSet(lhs.b_interval)
			else:
				A = BlessedSet(lhs)
				B = BlessedSet(sp.FiniteSet(0))

			if isinstance(rhs, ComplexRegion) or rhs is sp.Complexes:
				C = BlessedSet(rhs.a_interval)
				D = BlessedSet(rhs.b_interval)
			else:
				C = BlessedSet(rhs)
				D = BlessedSet(sp.FiniteSet(0))

			# + | -: Seperable
			## -> Recursively rely on FiniteSet|Range|Interval implementation.
			if op in [operator.add, operator.sub]:
				nset = BlessedSet.reals_to_complex(
					op(A, C),
					op(B, D),
				).bset

			# *: Standard Arithmetic Rules
			## -> Recursively rely on FiniteSet|Range|Interval implementation.
			elif op is operator.mul:
				nset = BlessedSet.reals_to_complex(
					A * C - B * D,
					B * C + A * D,
				).bset

			# /: Standard Arithmetic Rules
			## -> Recursively rely on FiniteSet|Range|Interval implementation.
			elif op is operator.truediv:
				denominator = C**2 + D**2
				nset = BlessedSet.reals_to_complex(
					(A * C + B * D) / denominator,
					(B * C + A * D) / denominator,
				).bset

			# **: Complex Exponentiation
			## -> May I just say, "oh boy"...
			## -> Decidedly more manual than the others...
			if op is operator.pow:
				# Account for FiniteSet
				## -> Must be done manually; SetExpr just gives up.
				if isinstance(rhs, sp.FiniteSet):
					fs = rhs
					nset = functools.reduce(
						lambda l, r: l | r,  # noqa: E741
						(self._operate_scalar(op, fs_el) for fs_el in fs),
					)

				# Account for ComplexRegion/Complexes | Complex
				## -> This is an INVOLVED calculation.
				## -> So, we give up.
				elif sp.FiniteSet(0) != D:
					log.error(
						'Exponentiation of set w/complex set is not (yet) supported; falling back to the entire set of complex numbers'
					)
					nset = sp.Complexes

				else:
					pos_C = abs(C) & sp.Interval(0, sp.oo)
					neg_C = abs(C * -1) & sp.Interval(0, sp.oo)

					halves = []
					for half_C in [neg_C, pos_C]:
						if half_C is sp.EmptySet:
							halves.append(sp.EmptySet)

						# Complex | Integer
						## -> Apply De Moivre's Formula
						elif half_C.issubset(sp.Integers):
							N = half_C
							r_N = abs(self) ** N
							arg_N = N**self.arg

							nset_half = BlessedSet.reals_to_complex(
								r_N * arg_N.cos,
								r_N * arg_N.sin,
							).bset

						# Complex | Reals
						elif half_C.issubset(sp.Rationals) or half_C.issubset(sp.Reals):
							log.error(
								'Exponentiation of set w/real or rational set is not (yet) supported; falling back to the entire set of complex numbers'
							)
							return BlessedSet(sp.Complexes)

						halves.append(nset_half)

					# Determine Reciprocal
					nset = (
						BlessedSet(halves[0]).reciprocal | BlessedSet(halves[1])
					).bset

		# Points
		## -> (FiniteSet | *) or (* | FiniteSet)
		## -> We've eliminated EmptySet, Union, and Complexes|ComplexRegion.
		## -> We're left to account for FiniteSet, Range, Interval.
		elif isinstance(lhs, sp.FiniteSet) or isinstance(rhs, sp.FiniteSet):
			fs = lhs if isinstance(lhs, sp.FiniteSet) else rhs
			other = rhs if isinstance(lhs, sp.FiniteSet) else lhs

			if isinstance(other, sp.FiniteSet):
				nset = set_expr_op(op, fs, other)

			elif isinstance(other, sp.Range):
				rng = other
				nset = functools.reduce(
					lambda l, r: l | r,  # noqa: E741
					(
						sp.Range(op(rng.inf, fs_el), op(rng.inf, fs_el) + 1, 1)
						for fs_el in fs
					),
				)

			elif isinstance(other, sp.Interval):
				nset = set_expr_op(op, lhs, rhs)

			else:
				raise NotImplementedError

		# Region

		## -> (Range | *) or (* | Range)
		## -> Eliminated EmptySet, Union, Complexes|ComplexRegion, FiniteSet.
		## -> We've left to account for Range, Interval.
		elif isinstance(lhs, sp.Range) or isinstance(rhs, sp.Range):
			rng = lhs if isinstance(lhs, sp.Range) else rhs
			other = rhs if isinstance(lhs, sp.Range) else lhs

			if isinstance(other, sp.Range):
				## -> Bound scaling is valid for +, -, *, /, **.
				start, _, _, stop = sorted(
					[
						op(lhs.start, rhs.start),
						op(lhs.start, rhs.stop),
						op(lhs.stop, rhs.start),
						op(lhs.stop, rhs.stop),
					]
				)
				nset = sp.Range(start, stop, 1)

			elif isinstance(other, sp.Interval):
				## -> Bound scaling is valid for +, -, *, /, **.
				itv = other
				(start, start_open), _, _, (end, end_open) = sorted(
					[
						(op(rng.start, itv.start), itv.left_open),
						(op(rng.start, itv.end), itv.right_open),
						(op(rng.stop, itv.start), itv.left_open),
						(op(rng.stop, itv.end), itv.right_open),
					],
					key=lambda el: el[0],
				)
				nset = sp.Interval(
					start,
					end,
					left_open=start_open,
					right_open=end_open,
				)

			else:
				raise NotImplementedError

		## -> Interval | Interval
		## -> Eliminated everything but Interval.
		elif isinstance(lhs, sp.Interval) and isinstance(rhs, sp.Interval):
			# Edge Case: (0,oo) * (0,oo)
			if (
				op is operator.mul
				and lhs.inf == 0
				and rhs.inf == 0
				and lhs.sup == sp.oo
				and rhs.sup == sp.oo
			):
				nset = sp.Interval(0, sp.oo, left_open=lhs.left_open or rhs.left_open)
			elif op is not operator.pow:
				nset = set_expr_op(op, lhs, rhs)
			else:
				(start, start_open), _, _, (end, end_open) = sorted(
					[
						(lhs.start**rhs.start, lhs.left_open & rhs.left_open),
						(lhs.start**rhs.end, lhs.left_open & rhs.right_open),
						(lhs.end**rhs.start, lhs.right_open & rhs.left_open),
						(lhs.end**rhs.end, lhs.right_open & rhs.right_open),
					],
					key=lambda el: el[0],
				)
				nset = sp.Interval(
					start.n(),
					end.n(),
					left_open=start_open,
					right_open=end_open,
				)
		else:
			raise NotImplementedError

		log.critical(['MINKOWSKI DONE', nset])
		return BlessedSet(nset)

	####################
	# - Operator Overload Dispatch
	####################
	def __add__(self, other: Scalar | typ.Self | sp.Set | set) -> typ.Self:
		"""Deduce the `BlessedSet` resulting from its element-wise addition with a scalar or another `BlessedSet`."""
		if isinstance(other, BlessedSet | sp.Set | set):
			return self._operate_minkowski(operator.add, BlessedSet(other))
		return self._operate_scalar(operator.add, other)

	def __radd__(self, other: Scalar | typ.Self) -> typ.Self:
		return other + self

	def __sub__(self, other: Scalar | typ.Self) -> typ.Self:
		"""Deduce the `BlessedSet` resulting from its element-wise subtraction with a scalar or another `BlessedSet`."""
		if isinstance(other, BlessedSet | sp.Set | set):
			return self._operate_minkowski(operator.sub, BlessedSet(other))
		return self._operate_scalar(operator.sub, other)

	def __rsub__(self, other: Scalar | typ.Self) -> typ.Self:
		return other - self

	def __mul__(self, other: Scalar | typ.Self) -> typ.Self:
		"""Deduce the `BlessedSet` resulting from its element-wise multiplication with a scalar or another `BlessedSet`."""
		if isinstance(other, BlessedSet | sp.Set | set):
			return self._operate_minkowski(operator.mul, BlessedSet(other))
		return self._operate_scalar(operator.mul, other)

	def __rmul__(self, other: Scalar | typ.Self) -> typ.Self:
		return other * self

	def __truediv__(self, other: Scalar | typ.Self) -> typ.Self:
		"""Deduce the `BlessedSet` resulting from its element-wise division with a scalar or another `BlessedSet`."""
		if isinstance(other, BlessedSet):
			return self._operate_minkowski(operator.truediv, BlessedSet(other))
		return self._operate_scalar(operator.truediv, other)

	def __rtruediv__(self, other: Scalar | typ.Self) -> typ.Self:
		return other / self

	def __pow__(self, other: Scalar | typ.Self) -> typ.Self:
		"""Deduce the `BlessedSet` resulting from its element-wise exponentiation with a scalar or another `BlessedSet`."""
		if isinstance(other, BlessedSet | sp.Set | set):
			return self._operate_minkowski(operator.pow, BlessedSet(other))
		return self._operate_scalar(operator.pow, other)

	def __rpow__(self, other: Scalar | typ.Self) -> typ.Self:
		return other**self

	def atan2(self, other: Scalar | typ.Self) -> typ.Self:
		lhs = self.bset
		if isinstance(other, BlessedSet | sp.Set | set):
			rhs = BlessedSet(other)
		else:
			rhs = other

		if not isinstance(lhs, ComplexRegion) and not isinstance(rhs, ComplexRegion):
			return BlessedSet.reals_to_complex(lhs, rhs).arg

		raise NotImplementedError
