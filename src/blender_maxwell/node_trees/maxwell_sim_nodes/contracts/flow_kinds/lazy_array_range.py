import dataclasses
import functools
import typing as typ
from types import MappingProxyType

import jax.numpy as jnp
import jaxtyping as jtyp
import sympy as sp

from blender_maxwell.utils import extra_sympy_units as spux

from .array import ArrayFlow
from .flow_kinds import FlowKind
from .lazy_value_func import LazyValueFuncFlow


@dataclasses.dataclass(frozen=True, kw_only=True)
class LazyArrayRangeFlow:
	r"""Represents a linearly/logarithmically spaced array using symbolic boundary expressions, with support for units and lazy evaluation.

	# Advantages
	Whenever an array can be represented like this, the advantages over an `ArrayFlow` are numerous.

	## Memory
	`ArrayFlow` generally has a memory scaling of $O(n)$.
	Naturally, `LazyArrayRangeFlow` is always constant, since only the boundaries and steps are stored.

	## Symbolic
	Both boundary points are symbolic expressions, within which pre-defined `sp.Symbol`s can participate in a constrained manner (ex. an integer symbol).

	One need not know the value of the symbols immediately - such decisions can be deferred until later in the computational flow.

	## Performant Unit-Aware Operations
	While `ArrayFlow`s are also unit-aware, the time-cost of _any_ unit-scaling operation scales with $O(n)$.
	`LazyArrayRangeFlow`, by contrast, scales as $O(1)$.

	As a result, more complicated operations (like symbolic or unit-based) that might be difficult to perform interactively in real-time on an `ArrayFlow` will work perfectly with this object, even with added complexity

	## High-Performance Composition and Gradiant
	With `self.as_func`, a `jax` function is produced that generates the array according to the symbolic `start`, `stop` and `steps`.
	There are two nice things about this:

	- **Gradient**: The gradient of the output array, with respect to any symbols used to define the input bounds, can easily be found using `jax.grad` over `self.as_func`.
	- **JIT**: When `self.as_func` is composed with other `jax` functions, and `jax.jit` is run to optimize the entire thing, the "cost of array generation" _will often be optimized away significantly or entirely_.

	Thus, as part of larger computations, the performance properties of `LazyArrayRangeFlow` is extremely favorable.

	## Numerical Properties
	Since the bounds support exact (ex. rational) calculations and symbolic manipulations (_by virtue of being symbolic expressions_), the opportunities for certain kinds of numerical instability are mitigated.

	Attributes:
		start: An expression generating a scalar, unitless, complex value for the array's lower bound.
			_Integer, rational, and real values are also supported._
		stop: An expression generating a scalar, unitless, complex value for the array's upper bound.
			_Integer, rational, and real values are also supported._
		steps: The amount of steps (**inclusive**) to generate from `start` to `stop`.
		scaling: The method of distributing `step` values between the two endpoints.
			Generally, the linear default is sufficient.

		unit: The unit of the generated array values

		symbols: Set of variables from which `start` and/or `stop` are determined.
	"""

	start: spux.ScalarUnitlessComplexExpr
	stop: spux.ScalarUnitlessComplexExpr
	steps: int
	scaling: typ.Literal['lin', 'geom', 'log'] = 'lin'

	unit: spux.Unit | None = None

	symbols: frozenset[spux.IntSymbol] = frozenset()

	@functools.cached_property
	def sorted_symbols(self) -> list[sp.Symbol]:
		"""Retrieves all symbols by concatenating int, real, and complex symbols, and sorting them by name.

		The order is guaranteed to be **deterministic**.

		Returns:
			All symbols valid for use in the expression.
		"""
		return sorted(self.symbols, key=lambda sym: sym.name)

	@functools.cached_property
	def mathtype(self) -> spux.MathType:
		# Get Start Mathtype
		if isinstance(self.start, spux.SympyType):
			start_mathtype = spux.MathType.from_expr(self.start)
		else:
			start_mathtype = spux.MathType.from_pytype(type(self.start))

		# Get Stop Mathtype
		if isinstance(self.stop, spux.SympyType):
			stop_mathtype = spux.MathType.from_expr(self.stop)
		else:
			stop_mathtype = spux.MathType.from_pytype(type(self.stop))

		# Check Equal
		if start_mathtype != stop_mathtype:
			return spux.MathType.combine(start_mathtype, stop_mathtype)

		return start_mathtype

	def __len__(self):
		return self.steps

	####################
	# - Units
	####################
	def correct_unit(self, corrected_unit: spux.Unit) -> typ.Self:
		"""Replaces the unit without rescaling the unitless bounds.

		Parameters:
			corrected_unit: The unit to replace the current unit with.

		Returns:
			A new `LazyArrayRangeFlow` with replaced unit.

		Raises:
			ValueError: If the existing unit is `None`, indicating that there is no unit to correct.
		"""
		if self.unit is not None:
			return LazyArrayRangeFlow(
				start=self.start,
				stop=self.stop,
				steps=self.steps,
				scaling=self.scaling,
				unit=corrected_unit,
				symbols=self.symbols,
			)

		msg = f'Tried to correct unit of unitless LazyDataValueRange "{corrected_unit}"'
		raise ValueError(msg)

	def rescale_to_unit(self, unit: spux.Unit) -> typ.Self:
		"""Replaces the unit, **with** rescaling of the bounds.

		Parameters:
			unit: The unit to convert the bounds to.

		Returns:
			A new `LazyArrayRangeFlow` with replaced unit.

		Raises:
			ValueError: If the existing unit is `None`, indicating that there is no unit to correct.
		"""
		if self.unit is not None:
			return LazyArrayRangeFlow(
				start=spux.scale_to_unit(self.start * self.unit, unit),
				stop=spux.scale_to_unit(self.stop * self.unit, unit),
				steps=self.steps,
				scaling=self.scaling,
				unit=unit,
				symbols=self.symbols,
			)

		msg = f'Tried to rescale unitless LazyDataValueRange to unit {unit}'
		raise ValueError(msg)

	def rescale_to_unit_system(self, unit_system: spux.Unit) -> typ.Self:
		"""Replaces the units, **with** rescaling of the bounds.

		Parameters:
			unit: The unit to convert the bounds to.

		Returns:
			A new `LazyArrayRangeFlow` with replaced unit.

		Raises:
			ValueError: If the existing unit is `None`, indicating that there is no unit to correct.
		"""
		if self.unit is not None:
			return LazyArrayRangeFlow(
				start=spux.strip_unit_system(
					spux.convert_to_unit_system(self.start * self.unit, unit_system),
					unit_system,
				),
				stop=spux.strip_unit_system(
					spux.convert_to_unit_system(self.start * self.unit, unit_system),
					unit_system,
				),
				steps=self.steps,
				scaling=self.scaling,
				unit=unit_system[spux.PhysicalType.from_unit(self.unit)],
				symbols=self.symbols,
			)

		msg = (
			f'Tried to rescale unitless LazyDataValueRange to unit system {unit_system}'
		)
		raise ValueError(msg)

	####################
	# - Bound Operations
	####################
	def rescale_bounds(
		self,
		rescale_func: typ.Callable[
			[spux.ScalarUnitlessComplexExpr], spux.ScalarUnitlessComplexExpr
		],
		reverse: bool = False,
	) -> typ.Self:
		"""Apply a function to the bounds, effectively rescaling the represented array.

		Notes:
			**It is presumed that the bounds are scaled with the same factor**.
			Breaking this presumption may have unexpected results.

			The scalar, unitless, complex-valuedness of the bounds must also be respected; additionally, new symbols must not be introduced.

		Parameters:
			scaler: The function that scales each bound.
			reverse: Whether to reverse the bounds after running the `scaler`.

		Returns:
			A rescaled `LazyArrayRangeFlow`.
		"""
		return LazyArrayRangeFlow(
			start=rescale_func(self.start if not reverse else self.stop),
			stop=rescale_func(self.stop if not reverse else self.start),
			steps=self.steps,
			scaling=self.scaling,
			unit=self.unit,
			symbols=self.symbols,
		)

	####################
	# - Lazy Representation
	####################
	@functools.cached_property
	def array_generator(
		self,
	) -> typ.Callable[
		[int | float | complex, int | float | complex, int],
		jtyp.Inexact[jtyp.Array, ' steps'],
	]:
		"""Compute the correct `jnp.*space` array generator, where `*` is one of the supported scaling methods.

		Returns:
			A `jax` function that takes a valid `start`, `stop`, and `steps`, and returns a 1D `jax` array.
		"""
		jnp_nspace = {
			'lin': jnp.linspace,
			'geom': jnp.geomspace,
			'log': jnp.logspace,
		}.get(self.scaling)
		if jnp_nspace is None:
			msg = f'ArrayFlow scaling method {self.scaling} is unsupported'
			raise RuntimeError(msg)

		return jnp_nspace

	@functools.cached_property
	def as_func(
		self,
	) -> typ.Callable[[int | float | complex, ...], jtyp.Inexact[jtyp.Array, ' steps']]:
		"""Create a function that can compute the non-lazy output array as a function of the symbols in the expressions for `start` and `stop`.

		Notes:
			The ordering of the symbols is identical to `self.symbols`, which is guaranteed to be a deterministically sorted list of symbols.

		Returns:
			A `LazyValueFuncFlow` that, given the input symbols defined in `self.symbols`,
		"""
		# Compile JAX Functions for Start/End Expressions
		## FYI, JAX-in-JAX works perfectly fine.
		start_jax = sp.lambdify(self.symbols, self.start, 'jax')
		stop_jax = sp.lambdify(self.symbols, self.stop, 'jax')

		# Compile ArrayGen Function
		def gen_array(
			*args: list[int | float | complex],
		) -> jtyp.Inexact[jtyp.Array, ' steps']:
			return self.array_generator(start_jax(*args), stop_jax(*args), self.steps)

		# Return ArrayGen Function
		return gen_array

	@functools.cached_property
	def as_lazy_value_func(self) -> LazyValueFuncFlow:
		"""Creates a `LazyValueFuncFlow` using the output of `self.as_func`.

		This is useful for ex. parameterizing the first array in the node graph, without binding an entire computed array.

		Notes:
			The the function enclosed in the `LazyValueFuncFlow` is identical to the one returned by `self.as_func`.

		Returns:
			A `LazyValueFuncFlow` containing `self.as_func`, as well as appropriate supporting settings.
		"""
		return LazyValueFuncFlow(
			func=self.as_func,
			func_args=[(spux.MathType.from_expr(sym)) for sym in self.symbols],
			supports_jax=True,
		)

	####################
	# - Realization
	####################
	def realize(
		self,
		symbol_values: dict[spux.Symbol, typ.Any] = MappingProxyType({}),
		kind: typ.Literal[FlowKind.Array, FlowKind.LazyValueFunc] = FlowKind.Array,
	) -> ArrayFlow | LazyValueFuncFlow:
		"""Apply a function to the bounds, effectively rescaling the represented array.

		Notes:
			**It is presumed that the bounds are scaled with the same factor**.
			Breaking this presumption may have unexpected results.

			The scalar, unitless, complex-valuedness of the bounds must also be respected; additionally, new symbols must not be introduced.

		Parameters:
			scaler: The function that scales each bound.
			reverse: Whether to reverse the bounds after running the `scaler`.

		Returns:
			A rescaled `LazyArrayRangeFlow`.
		"""
		if not set(self.symbols).issubset(set(symbol_values.keys())):
			msg = f'Provided symbols ({set(symbol_values.keys())}) do not provide values for all expression symbols ({self.symbols}) that may be found in the boundary expressions (start={self.start}, end={self.end})'
			raise ValueError(msg)

		# Realize Symbols
		realized_start = spux.sympy_to_python(
			self.start.subs({sym: symbol_values[sym.name] for sym in self.symbols})
		)
		realized_stop = spux.sympy_to_python(
			self.stop.subs({sym: symbol_values[sym.name] for sym in self.symbols})
		)

		# Return Linspace / Logspace
		def gen_array() -> jtyp.Inexact[jtyp.Array, ' steps']:
			return self.array_generator(realized_start, realized_stop, self.steps)

		if kind == FlowKind.Array:
			return ArrayFlow(values=gen_array(), unit=self.unit, is_sorted=True)
		if kind == FlowKind.LazyValueFunc:
			return LazyValueFuncFlow(func=gen_array, supports_jax=True)

		msg = f'Invalid kind: {kind}'
		raise TypeError(msg)

	@functools.cached_property
	def realize_array(self) -> ArrayFlow:
		return self.realize()
