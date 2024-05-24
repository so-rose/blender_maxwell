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

import dataclasses
import enum
import functools
import typing as typ
from fractions import Fraction
from types import MappingProxyType

import jax.numpy as jnp
import jaxtyping as jtyp
import sympy as sp
import sympy.physics.units as spu

from blender_maxwell.utils import extra_sympy_units as spux
from blender_maxwell.utils import logger, sim_symbols

from .array import ArrayFlow

log = logger.get(__name__)


class ScalingMode(enum.StrEnum):
	"""Identifier for how to space steps between two boundaries.

	Attributes:
		Lin: Uniform spacing between two endpoints.
		Geom: Log spacing between two endpoints, given as values.
		Log: Log spacing between two endpoints, given as powers of a common base.
	"""

	Lin = enum.auto()
	Geom = enum.auto()
	Log = enum.auto()

	@staticmethod
	def to_name(v: typ.Self) -> str:
		SM = ScalingMode
		return {
			SM.Lin: 'Linear',
			SM.Geom: 'Geometric',
			SM.Log: 'Logarithmic',
		}[v]

	@staticmethod
	def to_icon(_: typ.Self) -> str:
		return ''


@dataclasses.dataclass(frozen=True, kw_only=True)
class RangeFlow:
	r"""Represents a finite spaced array using symbolic boundary expressions.

	Whenever an array can be represented like this, the advantages over an `ArrayFlow` are numerous.

	# Memory Scaling
	`ArrayFlow` generally has a memory scaling of $O(n)$.
	Naturally, `RangeFlow` is always constant, since only the boundaries and steps are stored.

	# Symbolic Bounds
	`self.start` and `self.stop` boundary points are symbolic expressions, within which any element of `self.symbols` can participate.

	**It is the user's responsibility** to ensure that `self.start < self.stop`.

	# Numerical Properties
	Since the bounds support exact (ex. rational) calculations and symbolic manipulations (_by virtue of being symbolic expressions_), the opportunities for certain kinds of numerical instability are mitigated.

	Attributes:
		start: An expression representing the unitless part of the finite, scalar, complex value for the array's lower bound.
			_Integer, rational, and real values are also supported._
		start: An expression representing the unitless part of the finite, scalar, complex value for the array's upper bound.
			_Integer, rational, and real values are also supported._
		steps: The amount of steps (**inclusive**) to generate from `start` to `stop`.
		scaling: The method of distributing `step` values between the two endpoints.

		unit: The unit to interpret the values as.

		symbols: Set of variables from which `start` and/or `stop` are determined.
	"""

	start: spux.ScalarUnitlessComplexExpr
	stop: spux.ScalarUnitlessComplexExpr
	steps: int = 0
	scaling: ScalingMode = ScalingMode.Lin

	unit: spux.Unit | None = None

	symbols: frozenset[sim_symbols.SimSymbol] = frozenset()

	# Helper Attributes
	pre_fourier_ideal_midpoint: spux.ScalarUnitlessComplexExpr | None = None

	####################
	# - SimSymbol Interop
	####################
	@staticmethod
	def from_sym(
		sym: sim_symbols.SimSymbol,
		steps: int = 50,
		scaling: ScalingMode | str = ScalingMode.Lin,
	) -> typ.Self:
		if sym.domain.start.is_infinite or sym.domain.end.is_infinite:
			use_steps = 0
		else:
			use_steps = steps

		return RangeFlow(
			start=sym.domain.start if sym.domain.start.is_finite else sp.S(-1),
			stop=sym.domain.end if sym.domain.end.is_finite else sp.S(1),
			steps=use_steps,
			scaling=ScalingMode(scaling),
			unit=sym.unit,
		)

	def to_sym(
		self,
		sym_name: sim_symbols.SimSymbolName,
	) -> typ.Self:
		physical_type = spux.PhysicalType.from_unit(self.unit, optional=True)

		return sim_symbols.SimSymbol(
			sym_name=sym_name,
			mathtype=self.mathtype,
			physical_type=(
				physical_type
				if physical_type is not None
				else spux.PhysicalType.NonPhysical
			),
			unit=self.unit,
			rows=1,
			cols=1,
		).set_domain(start=self.realize_start(), end=self.realize_end())

	####################
	# - Symbols
	####################
	@functools.cached_property
	def sorted_symbols(self) -> list[sim_symbols.SimSymbol]:
		"""Retrieves all symbols by concatenating int, real, and complex symbols, and sorting them by name.

		The order is guaranteed to be **deterministic**.

		Returns:
			All symbols valid for use in the expression.
		"""
		return sorted(self.symbols, key=lambda sym: sym.name)

	@functools.cached_property
	def sorted_sp_symbols(self) -> list[spux.Symbol]:
		"""Computes `sympy` symbols from `self.sorted_symbols`.

		Returns:
			All symbols valid for use in the expression.
		"""
		return [sym.sp_symbol for sym in self.sorted_symbols]

	####################
	# - Properties
	####################
	@functools.cached_property
	def unit_factor(self) -> spux.SympyExpr:
		return self.unit if self.unit is not None else sp.S(1)

	def __len__(self) -> int:
		"""Compute the length of the array that would be realized.

		Returns:
			The number of steps.
		"""
		return self.steps

	@functools.cached_property
	def mathtype(self) -> spux.MathType:
		"""Conservatively compute the most stringent `spux.MathType` that can represent both `self.start` and `self.stop`.

		Notes:
			The mathtype is determined from start/stop either using `sympy` assumptions, or as Python types.

			For precise information on how start/stop are "combined", see `spux.MathType.combine()`.

		Returns:
			All symbols valid for use in the expression.
		"""
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
		combined_mathtype = spux.MathType.combine(start_mathtype, stop_mathtype)
		log.debug(
			'%s: Computed MathType as %s (start_mathtype=%s, stop_mathtype=%s)',
			self,
			combined_mathtype,
			start_mathtype,
			stop_mathtype,
		)
		return combined_mathtype

	####################
	# - Methods
	####################
	@property
	def ideal_midpoint(self) -> spux.SympyExpr:
		return (self.stop + self.start) / 2

	@property
	def ideal_range(self) -> spux.SympyExpr:
		return self.stop - self.start

	def rescale(
		self, rescale_func, reverse: bool = False, new_unit: spux.Unit | None = None
	) -> typ.Self:
		"""Apply an order-preserving function to each bound, then (optionally) transform the result w/new unit and/or order.

		An optimized expression will be built and applied to `self.values` using `sympy.lambdify()`.

		Parameters:
			rescale_func: An **order-preserving** function to apply to each array element.
			reverse: Whether to reverse the order of the result.
			new_unit: An (optional) new unit to scale the result to.
		"""
		new_pre_start = self.start if not reverse else self.stop
		new_pre_stop = self.stop if not reverse else self.start

		new_start = rescale_func(new_pre_start * self.unit_factor)
		new_stop = rescale_func(new_pre_stop * self.unit_factor)

		return RangeFlow(
			start=(
				spux.scale_to_unit(new_start, new_unit)
				if new_unit is not None
				else new_start
			),
			stop=(
				spux.scale_to_unit(new_stop, new_unit)
				if new_unit is not None
				else new_stop
			),
			steps=self.steps,
			scaling=self.scaling,
			unit=new_unit,
			symbols=self.symbols,
		)

	def nearest_idx_of(self, value: spux.SympyType, require_sorted: bool = True) -> int:
		raise NotImplementedError

	@functools.cached_property
	def bound_fourier_transform(self):
		r"""Treat this `RangeFlow` it as an axis along which a fourier transform is being performed, such that its bounds scale according to the Nyquist Limit.

		# Sampling Theory
		In general, the Fourier Transform is an operator that works on infinite, periodic, continuous functions.
		In this context alone it is a ideal transform (in terms of information retention), one which degrades quite gracefully in the face of practicalities like windowing (allowing them to apply analytically to non-periodic functions too).
		While often used to transform an axis of time to an axis of frequency, in general this transform "simply" extracts all repeating structures from a function.
		This is illustrated beautifully in the way that the output unit becomes the reciprocal of the input unit, which is the theory underlying why we say that measurements recieved as a reciprocal unit are in a "reciprocal space" (also called "k-space").

		The real world is not so nice, of course, and as such we must generally make do with the Discrete Fourier Transform.
		Even with bounded discrete information, we can annoy many mathematicians by defining a DFT in such a way that "structure per thing" ($\frac{1}{\texttt{unit}}$) still makes sense to us (to them, maybe not).
		A DFT can still only retain the information given to it, but so long as we have enough "original structure", any "repeating structure" should be extractable with sufficient clarity to be useful.

		What "sufficient clarity" means is the basis for the entire field of "sampling theory".
		The theoretical maximum for the "fineness of repetition" that is "noticeable" in the fourier-transformed of some data is characterized by a theoretical upper bound called the Nyquist Frequency / Limit, which ends up being half of the sampling rate.
		Thus, to determine bounds on the data, use of the Nyquist Limit is generally a good starting point.

		Of course, when the discrete data comes from a discretization of a continuous signal, information from higher frequencies might still affect the discrete results.
		They do little else than cause havoc, though - best causing noise, and at worst causing structured artifacts (sometimes called "aliasing").
		Some of the first innovations in sampling theory were related to "anti-aliasing" filters, whose sole purpose is to try to remove informational frequencies above the Nyquist Limit of the discrete sensor.

		In FDTD simulation, we're generally already ahead when it comes to aliasing, since our field values come from an already-discrete process.
		That is, unless we start overly "binning" (averaging over $n$ discrete observations); in this case, care should be taken to make sure that interesting results aren't merely unfortunately structured aliasing artifacts.

		For more, see <https://en.wikipedia.org/wiki/Nyquist%E2%80%93Shannon_sampling_theorem>

		# Implementation
		In practice, our goal in `RangeFlow` is to compute the bounds of the index array along the fourier-transformed data axis.
		The reciprocal of the unit will be taken (when unitless, `1/1=`).
		The raw Nyquist Limit $n_q$ will be used to bound the unitless part of the output as $[-n_q, n_q]$

		Raises:
			ValueError: If `self.scaling` is not linear, since the FT can only be performed on uniformly spaced data.
		"""
		if self.scaling is ScalingMode.Lin:
			nyquist_limit = self.steps / self.ideal_range

			# Return New Bounds w/Nyquist Theorem
			## -> The Nyquist Limit describes "max repeated info per sample".
			## -> Information can still record "faster" than the Nyquist Limit.
			## -> It will just be either noise (best case), or banded artifacts.
			## -> This is called "aliasing", and it's best to try and filter it.
			## -> Sims generally "bin" yee cells, which sacrifices some NyqLim.
			return RangeFlow(
				start=-nyquist_limit,
				stop=nyquist_limit,
				scaling=self.scaling,
				unit=1 / self.unit if self.unit is not None else None,
				pre_fourier_ideal_midpoint=self.ideal_midpoint,
			)

		msg = f'Cant fourier-transform an index array as a boundary, when the RangeArray has a non-linear bound {self.scaling}'
		raise ValueError(msg)

	@functools.cached_property
	def bound_inv_fourier_transform(self):
		r"""Treat this `RangeFlow` as an axis along which an inverse fourier transform is being performed, such that its Nyquist-limit bounds are transformed back into values along the original unit dimension.

		See `self.bound_fourier_transform` for the theoretical concepts.

		Notes:
			**The discrete inverse fourier transform always centers its output at $0$**.

			Of course, it's entirely probable that the original signal was not centered at $0$.
			For this reason, when performing a Fourier transform, `self.bound_fourier_transform` sets a special variable, `self.pre_fourier_ideal_midpoint`.
			When set, it will retain the `self.ideal_midpoint` around which both `self.start` and `self.stop` should be centered after an inverse FT.

			If `self.pre_fourier_ideal_midpoint` is set, then it will be used as the midpoint of the output's `start`/`stop`.
			Otherwise, $0$ will be used - in which case the user should themselves, manually, shift the output if needed.
		"""
		if self.scaling is ScalingMode.Lin:
			orig_ideal_range = self.steps / self.ideal_range

			orig_start_centered = -orig_ideal_range
			orig_stop_centered = orig_ideal_range
			orig_ideal_midpoint = (
				self.pre_fourier_ideal_midpoint
				if self.pre_fourier_ideal_midpoint is not None
				else sp.S(0)
			)

			# Return New Bounds w/Inverse of Nyquist Theorem
			return RangeFlow(
				start=-orig_start_centered + orig_ideal_midpoint,
				stop=orig_stop_centered + orig_ideal_midpoint,
				scaling=self.scaling,
				unit=1 / self.unit if self.unit is not None else None,
			)

		msg = f'Cant fourier-transform an index array as a boundary, when the RangeArray has a non-linear bound {self.scaling}'
		raise ValueError(msg)

	####################
	# - Exporters
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
			ScalingMode.Lin: jnp.linspace,
			ScalingMode.Geom: jnp.geomspace,
			ScalingMode.Log: jnp.logspace,
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
			The ordering of the symbols is identical to `self.sorted_symbols`, which is guaranteed to be a deterministically sorted list of symbols.

		Returns:
			A function that generates a 1D numerical array equivalent to the range represented in this `RangeFlow`.
		"""
		# Compile JAX Functions for Start/End Expressions
		## -> FYI, JAX-in-JAX works perfectly fine.
		start_jax = sp.lambdify(self.sorted_sp_symbols, self.start, 'jax')
		stop_jax = sp.lambdify(self.sorted_sp_symbols, self.stop, 'jax')

		# Compile ArrayGen Function
		def gen_array(
			*args: list[int | float | complex],
		) -> jtyp.Inexact[jtyp.Array, ' steps']:
			return self.array_generator(start_jax(*args), stop_jax(*args), self.steps)

		# Return ArrayGen Function
		return gen_array

	####################
	# - Realization
	####################
	def realize_symbols(
		self,
		symbol_values: dict[sim_symbols.SimSymbol, spux.SympyExpr] = MappingProxyType(
			{}
		),
	) -> dict[sp.Symbol, spux.ScalarUnitlessComplexExpr]:
		"""Realize **all** input symbols to the `RangeFlow`.

		Parameters:
			symbol_values: A scalar, unitless, complex `sympy` expression for each symbol defined in `self.symbols`.

		Returns:
			A dictionary directly usable in expression substitutions using `sp.Basic.subs()`.
		"""
		if self.symbols == set(symbol_values.keys()):
			realized_syms = {}
			for sym in self.sorted_symbols:
				sym_value = symbol_values[sym]

				# Sympy Expression
				## -> We need to conform the expression to the SimSymbol.
				## -> Mainly, this is
				if (
					isinstance(sym_value, spux.SympyType)
					and not isinstance(sym_value, sp.MatrixBase)
					and not spux.uses_units(sym_value)
				):
					v = sym.conform(sym_value)
				else:
					msg = f'RangeFlow: No realization support for symbolic value {sym_value} (type={type(sym_value)})'
					raise NotImplementedError(msg)

				realized_syms |= {sym: v}

		msg = f'RangeFlow: Not all symbols were given a value during realization (symbols={self.symbols}, symbol_values={symbol_values})'
		raise ValueError(msg)

	def realize_start(
		self,
		symbol_values: dict[sim_symbols.SimSymbol, spux.SympyExpr] = MappingProxyType(
			{}
		),
	) -> int | float | complex:
		"""Realize the start-bound by inserting particular values for each symbol."""
		realized_symbols = self.realize_symbols(symbol_values)
		return spux.sympy_to_python(self.start.subs(realized_symbols))

	def realize_stop(
		self,
		symbol_values: dict[sim_symbols.SimSymbol, spux.SympyExpr] = MappingProxyType(
			{}
		),
	) -> int | float | complex:
		"""Realize the stop-bound by inserting particular values for each symbol."""
		realized_symbols = self.realize_symbols(symbol_values)
		return spux.sympy_to_python(self.stop.subs(realized_symbols))

	def realize_step_size(
		self,
		symbol_values: dict[sim_symbols.SimSymbol, spux.SympyExpr] = MappingProxyType(
			{}
		),
	) -> int | float | complex:
		"""Realize the stop-bound by inserting particular values for each symbol."""
		if self.scaling is not ScalingMode.Lin:
			msg = 'Non-linear scaling mode not yet suported'
			raise NotImplementedError(msg)

		raw_step_size = (
			self.realize_stop(symbol_values) - self.realize_start(symbol_values) + 1
		) / self.steps

		if self.mathtype is spux.MathType.Integer and raw_step_size.is_integer():
			return int(raw_step_size)
		return raw_step_size

	def realize(
		self,
		symbol_values: dict[sim_symbols.SimSymbol, spux.SympyExpr] = MappingProxyType(
			{}
		),
	) -> ArrayFlow:
		"""Realize the array represented by this `RangeFlow` by realizing each bound, then generating all intermediate values as an array.

		Parameters:
			symbol_values: The particular values for each symbol, which will be inserted into the expression of each bound to realize them.

		Returns:
			An `ArrayFlow` containing this realized `RangeFlow`.
		"""
		## TODO: Check symbol values for coverage.

		return ArrayFlow(
			values=self.as_func(
				*[
					spux.scale_to_unit_system(symbol_values[sym])
					for sym in self.sorted_symbols
				]
			),
			unit=self.unit,
			is_sorted=True,
		)

	@functools.cached_property
	def realize_array(self) -> ArrayFlow:
		"""Standardized access to `self.realize()` when there are no symbols.

		Raises:
			ValueError: If there are symbols defined in `self.symbols`.
		"""
		if not self.symbols:
			return self.realize()

		msg = f'RangeFlow: Cannot use ".realize_array" when symbols are defined (symbols={self.symbols}, RangeFlow={self}'
		raise ValueError(msg)

	@property
	def values(self) -> jtyp.Inexact[jtyp.Array, '...']:
		"""Alias for `realize_array.values`."""
		return self.realize_array.values

	def __getitem__(self, subscript: slice):
		"""Implement indexing and slicing in a sane way.

		- **Integer Index**: Not yet implemented.
		- **Slice**: Return the `RangeFlow` that creates the same `ArrayFlow` as would be created by computing `self.realize_array`, then slicing that.
		"""
		if isinstance(subscript, slice) and self.scaling == ScalingMode.Lin:
			# Parse Slice
			start = subscript.start if subscript.start is not None else 0
			stop = subscript.stop if subscript.stop is not None else self.steps
			step = subscript.step if subscript.step is not None else 1

			slice_steps = (stop - start + step - 1) // step

			# Compute New Start/Stop
			step_size = self.realize_step_size()
			new_start = step_size * start
			new_stop = new_start + step_size * slice_steps

			return RangeFlow(
				start=sp.S(new_start),
				stop=sp.S(new_stop),
				steps=slice_steps,
				scaling=self.scaling,
				unit=self.unit,
				symbols=self.symbols,
			)

		raise NotImplementedError

	####################
	# - Units
	####################
	def correct_unit(self, corrected_unit: spux.Unit) -> typ.Self:
		"""Replaces the unit without rescaling the unitless bounds.

		Parameters:
			corrected_unit: The unit to replace the current unit with.

		Returns:
			A new `RangeFlow` with replaced unit.

		Raises:
			ValueError: If the existing unit is `None`, indicating that there is no unit to correct.
		"""
		return RangeFlow(
			start=self.start,
			stop=self.stop,
			steps=self.steps,
			scaling=self.scaling,
			unit=corrected_unit,
			symbols=self.symbols,
		)

	def rescale_to_unit(self, unit: spux.Unit) -> typ.Self:
		"""Replaces the unit, **with** rescaling of the bounds.

		Parameters:
			unit: The unit to convert the bounds to.

		Returns:
			A new `RangeFlow` with replaced unit.

		Raises:
			ValueError: If the existing unit is `None`, indicating that there is no unit to correct.
		"""
		if self.unit is not None:
			return RangeFlow(
				start=spux.scale_to_unit(self.start * self.unit, unit),
				stop=spux.scale_to_unit(self.stop * self.unit, unit),
				steps=self.steps,
				scaling=self.scaling,
				unit=unit,
				symbols=self.symbols,
			)
		return RangeFlow(
			start=self.start * unit,
			stop=self.stop * unit,
			steps=self.steps,
			scaling=self.scaling,
			unit=unit,
			symbols=self.symbols,
		)

	def rescale_to_unit_system(
		self, unit_system: spux.UnitSystem | None = None
	) -> typ.Self:
		"""Replaces the units, **with** rescaling of the bounds.

		Parameters:
			unit: The unit to convert the bounds to.

		Returns:
			A new `RangeFlow` with replaced unit.

		Raises:
			ValueError: If the existing unit is `None`, indicating that there is no unit to correct.
		"""
		return RangeFlow(
			start=spux.scale_to_unit_system(self.start * self.unit, unit_system),
			stop=spux.scale_to_unit_system(self.stop * self.unit, unit_system),
			steps=self.steps,
			scaling=self.scaling,
			unit=spux.convert_to_unit_system(self.unit, unit_system),
			symbols=self.symbols,
		)
