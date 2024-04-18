import dataclasses
import enum
import functools
import typing as typ
from types import MappingProxyType

import jax
import jax.numpy as jnp
import numba
import sympy as sp
import sympy.physics.units as spu
import typing_extensions as typx

from blender_maxwell.utils import extra_sympy_units as spux

from .socket_types import SocketType


class FlowKind(enum.StrEnum):
	"""Defines a kind of data that can flow between nodes.

	Each node link can be thought to contain **multiple pipelines for data to flow along**.
	Each pipeline is cached incrementally, and independently, of the others.
	Thus, the same socket can easily support several kinds of related data flow at the same time.

	Attributes:
		Capabilities: Describes a socket's linkeability with other sockets.
			Links between sockets with incompatible capabilities will be rejected.
			This doesn't need to be defined normally, as there is a default.
			However, in some cases, defining it manually to control linkeability more granularly may be desirable.
		Value: A generic object, which is "directly usable".
			This should be chosen when a more specific flow kind doesn't apply.
		Array: An object with dimensions, and possibly a unit.
			Whenever a `Value` is defined, a single-element `list` will also be generated by default as `Array`
			However, for any other array-like variants (or sockets that only represent array-like objects), `Array` should be defined manually.
		LazyValueFunc: A composable function.
			Can be used to represent computations for which all data is not yet known, or for which just-in-time compilation can drastically increase performance.
		LazyArrayRange: An object that generates an `Array` from range information (start/stop/step/spacing).
			This should be used instead of `Array` whenever possible.
		Param: A dictionary providing particular parameters for a lazy value.
		Info: An dictionary providing extra context about any aspect of flow.
	"""

	Capabilities = enum.auto()

	# Values
	Value = enum.auto()
	Array = enum.auto()

	# Lazy
	LazyValue = enum.auto()
	LazyArrayRange = enum.auto()

	# Auxiliary
	Params = enum.auto()
	Info = enum.auto()

	@classmethod
	def scale_to_unit_system(cls, kind: typ.Self, value, socket_type, unit_system):
		if kind == cls.Value:
			return spux.sympy_to_python(
				spux.scale_to_unit(
					value,
					unit_system[socket_type],
				)
			)
		if kind == cls.LazyArrayRange:
			return value.rescale_to_unit(unit_system[socket_type])

		msg = 'Tried to scale unknown kind'
		raise ValueError(msg)


####################
# - Capabilities
####################
@dataclasses.dataclass(frozen=True, kw_only=True)
class CapabilitiesFlow:
	socket_type: SocketType
	active_kind: FlowKind

	is_universal: bool = False

	def is_compatible_with(self, other: typ.Self) -> bool:
		return (
			self.socket_type == other.socket_type
			and self.active_kind == other.active_kind
		) or other.is_universal


####################
# - Value
####################
ValueFlow: typ.TypeAlias = typ.Any


####################
# - Value Array
####################
@dataclasses.dataclass(frozen=True, kw_only=True)
class ArrayFlow:
	"""A simple, flat array of values with an optionally-attached unit.

	Attributes:
		values: An ND array-like object of arbitrary numerical type.
		unit: A `sympy` unit.
			None if unitless.
	"""

	values: jax.Array
	unit: spu.Quantity | None = None

	def correct_unit(self, real_unit: spu.Quantity) -> typ.Self:
		if self.unit is not None:
			return ArrayFlow(values=self.values, unit=real_unit)

		msg = f'Tried to correct unit of unitless LazyDataValueRange "{real_unit}"'
		raise ValueError(msg)

	def rescale_to_unit(self, unit: spu.Quantity) -> typ.Self:
		if self.unit is not None:
			return ArrayFlow(
				values=float(spux.scaling_factor(self.unit, unit)) * self.values,
				unit=unit,
			)
			## TODO: Is this scaling numerically stable?

		msg = f'Tried to rescale unitless LazyDataValueRange to unit {unit}'
		raise ValueError(msg)


####################
# - Lazy Value Func
####################
LazyFunction: typ.TypeAlias = typ.Callable[[typ.Any, ...], ValueFlow]


@dataclasses.dataclass(frozen=True, kw_only=True)
class LazyValueFuncFlow:
	r"""Encapsulates a lazily evaluated data value as a composable function with bound and free arguments.

	- **Bound Args**: Arguments that are realized when **defining** the lazy value.
		Both positional values and keyword values are supported.
	- **Free Args**: Arguments that are specified when evaluating the lazy value.
		Both positional values and keyword values are supported.

	The **root function** is encapsulated using `from_function`, and must accept arguments in the following order:

	$$
		f_0:\ \ \ \ (\underbrace{b_1, b_2, ...}_{\text{Bound}}\ ,\ \underbrace{r_1, r_2, ...}_{\text{Free}}) \to \text{output}_0
	$$

	Subsequent **composed functions** are encapsulated from the _root function_, and are created with `root_function.compose`.
	They must accept arguments in the following order:

	$$
		f_k:\ \ \ \ (\underbrace{b_1, b_2, ...}_{\text{Bound}}\ ,\ \text{output}_{k-1} ,\ \underbrace{r_p, r_{p+1}, ...}_{\text{Free}}) \to \text{output}_k
	$$

	Attributes:
		function: The function to be lazily evaluated.
		bound_args: Arguments that will be packaged into function, which can't be later modifier.
		func_kwargs: Arguments to be specified by the user at the time of use.
		supports_jax: Whether the contained `self.function` can be compiled with JAX's JIT compiler.
		supports_numba: Whether the contained `self.function` can be compiled with Numba's JIT compiler.
	"""

	func: LazyFunction
	func_kwargs: dict[str, type]
	supports_jax: bool = False
	supports_numba: bool = False

	@staticmethod
	def from_func(
		func: LazyFunction,
		supports_jax: bool = False,
		supports_numba: bool = False,
		**func_kwargs: dict[str, type],
	) -> typ.Self:
		return LazyValueFuncFlow(
			func=func,
			func_kwargs=func_kwargs,
			supports_jax=supports_jax,
			supports_numba=supports_numba,
		)

	# Composition
	def compose_within(
		self,
		enclosing_func: LazyFunction,
		supports_jax: bool = False,
		supports_numba: bool = False,
		**enclosing_func_kwargs: dict[str, type],
	) -> typ.Self:
		return LazyValueFuncFlow(
			function=lambda **kwargs: enclosing_func(
				self.func(**{k: v for k, v in kwargs if k in self.func_kwargs}),
				**kwargs,
			),
			func_kwargs=self.func_kwargs | enclosing_func_kwargs,
			supports_jax=self.supports_jax and supports_jax,
			supports_numba=self.supports_numba and supports_numba,
		)

	@functools.cached_property
	def func_jax(self) -> LazyFunction:
		if self.supports_jax:
			return jax.jit(self.func)

		msg = 'Can\'t express LazyValueFuncFlow as JAX function (using jax.jit), since "self.supports_jax" is False'
		raise ValueError(msg)

	@functools.cached_property
	def func_numba(self) -> LazyFunction:
		if self.supports_numba:
			return numba.jit(self.func)

		msg = 'Can\'t express LazyValueFuncFlow as Numba function (using numba.jit), since "self.supports_numba" is False'
		raise ValueError(msg)


####################
# - Lazy Array Range
####################
@dataclasses.dataclass(frozen=True, kw_only=True)
class LazyArrayRangeFlow:
	symbols: set[sp.Symbol]

	start: sp.Basic
	stop: sp.Basic
	steps: int
	scaling: typx.Literal['lin', 'geom', 'log'] = 'lin'

	unit: spu.Quantity | None = False

	def correct_unit(self, real_unit: spu.Quantity) -> typ.Self:
		if self.unit is not None:
			return LazyArrayRangeFlow(
				symbols=self.symbols,
				unit=real_unit,
				start=self.start,
				stop=self.stop,
				steps=self.steps,
				scaling=self.scaling,
			)

		msg = f'Tried to correct unit of unitless LazyDataValueRange "{real_unit}"'
		raise ValueError(msg)

	def rescale_to_unit(self, unit: spu.Quantity) -> typ.Self:
		if self.unit is not None:
			return LazyArrayRangeFlow(
				symbols=self.symbols,
				unit=unit,
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
		return LazyArrayRangeFlow(
			symbols=self.symbols,
			unit=self.unit,
			start=spu.convert_to(
				bound_cb(self.start if not reverse else self.stop), self.unit
			),
			stop=spu.convert_to(
				bound_cb(self.stop if not reverse else self.start), self.unit
			),
			steps=self.steps,
			scaling=self.scaling,
		)

	def realize(
		self, symbol_values: dict[sp.Symbol, ValueFlow] = MappingProxyType({})
	) -> ArrayFlow:
		# Realize Symbols
		if self.unit is None:
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
			return ArrayFlow(
				values=jnp.linspace(start, stop, self.steps), unit=self.unit
			)
		if self.scaling == 'geom':
			return ArrayFlow(jnp.geomspace(start, stop, self.steps), self.unit)
		if self.scaling == 'log':
			return ArrayFlow(jnp.logspace(start, stop, self.steps), self.unit)

		msg = f'ArrayFlow scaling method {self.scaling} is unsupported'
		raise RuntimeError(msg)


####################
# - Params
####################
ParamsFlow: typ.TypeAlias = dict[str, typ.Any]


####################
# - Lazy Value Func
####################
InfoFlow: typ.TypeAlias = dict[str, typ.Any]
