import dataclasses
import enum
import functools
import typing as typ
from types import MappingProxyType

# import colour  ## TODO
import numpy as np
import sympy as sp
import sympy.physics.units as spu
import typing_extensions as typx

from ....utils import extra_sympy_units as spux
from ....utils import sci_constants as constants
from .socket_types import SocketType


class DataFlowKind(enum.StrEnum):
	"""Defines a shape/kind of data that may flow through a node tree.

	Since a node socket may define one of each, we can support several related kinds of data flow through the same node-graph infrastructure.

	Attributes:
		Value: A value without any unknown symbols.
			- Basic types aka. float, int, list, string, etc. .
			- Exotic (immutable-ish) types aka. numpy array, KDTree, etc. .
			- A usable constructed object, ex. a `tidy3d.Box`.
			- Expressions (`sp.Expr`) that don't have unknown variables.
			- Lazy sequences aka. generators, with all data bound.
		SpectralValue: A value defined along a spectral range.
			- {`np.array`

		LazyValue: An object which, when given new data, can make many values.
			- An `sp.Expr`, which might need `simplify`ing, `jax` JIT'ing, unit cancellations, variable substitutions, etc. before use.
			- Lazy objects, for which all parameters aren't yet known.
			- A computational graph aka. `aesara`, which may even need to be handled before

		Capabilities: A `ValueCapability` object providing compatibility.

	# Value Data Flow
	Simply passing values is the simplest and easiest use case.

	This doesn't mean it's "dumb" - ex. a `sp.Expr` might, before use, have `simplify`, rewriting, unit cancellation, etc. run.
	All of this is okay, as long as there is no *introduction of new data* ex. variable substitutions.


	# Lazy Value Data Flow
	By passing (essentially) functions, one supports:
	- **Lightness**: While lazy values can be made expensive to construct, they will generally not be nearly as heavy to handle when trying to work with ex. operations on voxel arrays.
	- **Performance**: Parameterizing ex. `sp.Expr` with variables allows one to build very optimized functions, which can make ex. node graph updates very fast if the only operation run is the `jax` JIT'ed function (aka. GPU accelerated) generated from the final full expression.
	- **Numerical Stability**: Libraries like `aesara` build a computational graph, which can be automatically rewritten to avoid many obvious conditioning / cancellation errors.
	- **Lazy Output**: The goal of a node-graph may not be the definition of a single value, but rather, a parameterized expression for generating *many values* with known properties. This is especially interesting for use cases where one wishes to build an optimization step using nodes.


	# Capability Passing
	By being able to pass "capabilities" next to other kinds of values, nodes can quickly determine whether a given link is valid without having to actually compute it.


	# Lazy Parameter Value
	When using parameterized LazyValues, one may wish to independently pass parameter values through the graph, so they can be inserted into the final (cached) high-performance expression without.

	The advantage of using a different data flow would be changing this kind of value would ONLY invalidate lazy parameter value caches, which would allow an incredibly fast path of getting the value into the lazy expression for high-performance computation.

	Implementation TBD - though, ostensibly, one would have a "parameter" node which both would only provide a LazyValue (aka. a symbolic variable), but would also be able to provide a LazyParamValue, which would be a particular value of some kind (probably via the `value` of some other node socket).
	"""

	Capabilities = enum.auto()

	# Values
	Value = enum.auto()
	ValueArray = enum.auto()
	ValueSpectrum = enum.auto()

	# Lazy
	LazyValue = enum.auto()
	LazyValueRange = enum.auto()
	LazyValueSpectrum = enum.auto()


####################
# - Data Structures: Capabilities
####################
@dataclasses.dataclass(frozen=True, kw_only=True)
class DataCapabilities:
	socket_type: SocketType
	active_kind: DataFlowKind

	is_universal: bool = False

	def is_compatible_with(self, other: typ.Self) -> bool:
		return (
			self.socket_type == other.socket_type
			and self.active_kind == other.active_kind
		) or other.is_universal


####################
# - Data Structures: Non-Lazy
####################
DataValue: typ.TypeAlias = typ.Any


@dataclasses.dataclass(frozen=True, kw_only=True)
class DataValueArray:
	"""A simple, flat array of values with an optionally-attached unit.

	Attributes:
		values: A 1D array-like object of arbitrary numerical type.
		unit: A `sympy` unit.
			None if unitless.
	"""

	values: typ.Sequence[DataValue]
	unit: spu.Quantity | None


@dataclasses.dataclass(frozen=True, kw_only=True)
class DataValueSpectrum:
	"""A numerical representation of a spectral distribution.

	Attributes:
		wls: A 1D `numpy` float array of wavelength values.
		wls_unit: The unit of wavelengths, as length dimension.
		values: A 1D `numpy` float array of values corresponding to wavelength values.
		values_unit: The unit of the value, as arbitrary dimension.
		freqs_unit: The unit of the value, as arbitrary dimension.
	"""

	# Wavelength
	wls: np.array
	wls_unit: spu.Quantity

	# Value
	values: np.array
	values_unit: spu.Quantity

	# Frequency
	freqs_unit: spu.Quantity = spu.hertz

	@functools.cached_property
	def freqs(self) -> np.array:
		"""The spectral frequencies, computed from the wavelengths.

		Frequencies are NOT reversed, so as to preserve the by-index mapping to `DataValueSpectrum.values`.

		Returns:
			Frequencies, as a unitless `numpy` array.
				Use `DataValueSpectrum.wls_unit` to interpret this return value.
		"""
		unitless_speed_of_light = spux.sympy_to_python(
			spux.scale_to_unit(
				constants.vac_speed_of_light, (self.wl_unit / self.freq_unit)
			)
		)
		return unitless_speed_of_light / self.wls

	# TODO: Colour Library
	# def as_colour_sd(self) -> colour.SpectralDistribution:
	# """Returns the `colour` representation of this spectral distribution, ideal for plotting and colorimetric analysis."""
	# return colour.SpectralDistribution(data=self.values, domain=self.wls)


####################
# - Data Structures: Lazy
####################
@dataclasses.dataclass(frozen=True, kw_only=True)
class LazyDataValue:
	callback: typ.Callable[[...], [DataValue]]

	def realize(self, *args: list[DataValue]) -> DataValue:
		return self.callback(*args)


@dataclasses.dataclass(frozen=True, kw_only=True)
class LazyDataValueRange:
	symbols: set[sp.Symbol]

	start: sp.Basic
	stop: sp.Basic
	steps: int
	scaling: typx.Literal['lin', 'geom', 'log'] = 'lin'

	has_unit: bool = False

	def rescale_to_unit(self, unit: spu.Quantity) -> typ.Self:
		if self.has_unit:
			return LazyDataValueRange(
				symbols=self.symbols,
				has_unit=self.has_unit,
				start=spu.convert_to(self.start, unit),
				stop=spu.convert_to(self.stop, unit),
				steps=self.steps,
				scaling=self.scaling,
			)

		msg = f'Tried to rescale unitless LazyDataValueRange to unit {unit}'
		raise ValueError(msg)

	def rescale_bounds(
		self,
		bound_cb: typ.Callable[[sp.Expr], sp.Expr],
		reverse: bool = False,
	) -> typ.Self:
		"""Call a function on both bounds (start and stop), creating a new `LazyDataValueRange`."""
		return LazyDataValueRange(
			symbols=self.symbols,
			has_unit=self.has_unit,
			start=bound_cb(self.start if not reverse else self.stop),
			stop=bound_cb(self.stop if not reverse else self.start),
			steps=self.steps,
			scaling=self.scaling,
		)

	def realize(
		self, symbol_values: dict[sp.Symbol, DataValue] = MappingProxyType({})
	) -> DataValueArray:
		# Realize Symbols
		if not self.has_unit:
			start = spux.sympy_to_python(self.start.subs(symbol_values))
			stop = spux.sympy_to_python(self.stop.subs(symbol_values))
		else:
			start = spux.sympy_to_python(
				spux.scale_to_unit(self.start.subs(symbol_values), self.unit)
			)
			stop = spux.sympy_to_python(
				spux.scale_to_unit(self.stop.subs(symbol_values), self.unit)
			)

		# Return Linspace / Logspace
		if self.scaling == 'lin':
			return DataValueArray(
				values=np.linspace(start, stop, self.steps), unit=self.unit
			)
		if self.scaling == 'geom':
			return DataValueArray(np.geomspace(start, stop, self.steps), self.unit)
		if self.scaling == 'log':
			return DataValueArray(np.logspace(start, stop, self.steps), self.unit)

		raise NotImplementedError


@dataclasses.dataclass(frozen=True, kw_only=True)
class LazyDataValueSpectrum:
	wl_unit: spu.Quantity
	value_unit: spu.Quantity
	value_expr: sp.Expr

	symbols: tuple[sp.Symbol, ...] = ()
	freq_symbol: sp.Symbol = sp.Symbol('lamda')  # noqa: RUF009

	def rescale_to_unit(self, unit: spu.Quantity) -> typ.Self:
		raise NotImplementedError

	@functools.cached_property
	def as_func(self) -> typ.Callable[[DataValue, ...], DataValue]:
		"""Generates an optimized function for numerical evaluation of the spectral expression."""
		return sp.lambdify([self.freq_symbol, *self.symbols], self.value_expr)

	def realize(
		self, wl_range: DataValueArray, symbol_values: tuple[DataValue, ...]
	) -> DataValueSpectrum:
		r"""Realizes the parameterized spectral function as a numerical spectral distribution.

		Parameters:
			wl_range: The lazy wavelength range to build the concrete spectral distribution with.
			symbol_values: Numerical values for each symbol, in the same order as defined in `LazyDataValueSpectrum.symbols`.
				The wavelength symbol ($\lambda$ by default) always goes first.
				_This is used to call the spectral function using the output of `.as_func()`._

		Returns:
			The concrete, numerical spectral distribution.
		"""
		return DataValueSpectrum(
			wls=wl_range.values,
			wls_unit=self.wl_unit,
			values=self.as_func(*list(symbol_values.values())),
			values_unit=self.value_unit,
		)
