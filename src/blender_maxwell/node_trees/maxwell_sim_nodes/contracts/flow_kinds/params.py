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

import functools
import typing as typ
from fractions import Fraction
from types import MappingProxyType

import jaxtyping as jtyp
import pydantic as pyd
import sympy as sp

from blender_maxwell.utils import sympy_extra as spux
from blender_maxwell.utils import logger, sim_symbols

from .array import ArrayFlow
from .expr_info import ExprInfo
from .lazy_range import RangeFlow

log = logger.get(__name__)


class ParamsFlow(pyd.BaseModel):
	"""Retrieves all symbols by concatenating int, real, and complex symbols, and sorting them by name.

	Returns:
		All symbols valid for use in the expression.
	"""

	model_config = pyd.ConfigDict(frozen=True)

	arg_targets: list[sim_symbols.SimSymbol] = pyd.Field(default_factory=list)
	kwarg_targets: dict[str, sim_symbols.SimSymbol] = pyd.Field(default_factory=dict)

	func_args: list[spux.SympyExpr] = pyd.Field(default_factory=list)
	func_kwargs: dict[str, spux.SympyExpr] = pyd.Field(default_factory=dict)

	symbols: frozenset[sim_symbols.SimSymbol] = frozenset()
	realized_symbols: dict[
		sim_symbols.SimSymbol, spux.SympyExpr | RangeFlow | ArrayFlow
	] = pyd.Field(default_factory=dict)

	####################
	# - Symbols
	####################
	@functools.cached_property
	def sorted_symbols(self) -> list[sim_symbols.SimSymbol]:
		"""Retrieves all symbols by concatenating int, real, and complex symbols, and sorting them by name.

		Returns:
			All symbols valid for use in the expression.
		"""
		return sorted(self.symbols, key=lambda sym: sym.name)

	@functools.cached_property
	def sorted_sp_symbols(self) -> list[sp.Symbol | sp.MatrixSymbol]:
		"""Computes `sympy` symbols from `self.sorted_symbols`.

		When the output is shaped, a single shaped symbol (`sp.MatrixSymbol`) is used to represent the symbolic name and shaping.
		This choice is made due to `MatrixSymbol`'s compatibility with `.lambdify` JIT.

		Returns:
			All symbols valid for use in the expression.
		"""
		return [sym.sp_symbol_matsym for sym in self.sorted_symbols]

	@functools.cached_property
	def all_sorted_symbols(self) -> list[sim_symbols.SimSymbol]:
		"""Retrieves all symbols by concatenating int, real, and complex symbols, and sorting them by name.

		Returns:
			All symbols valid for use in the expression.
		"""
		key_func = lambda sym: sym.name  # noqa: E731
		return sorted(self.symbols, key=key_func) + sorted(
			self.realized_symbols.keys(), key=key_func
		)

	@functools.cached_property
	def all_sorted_sp_symbols(self) -> list[sim_symbols.SimSymbol]:
		"""Retrieves all symbols by concatenating int, real, and complex symbols, and sorting them by name.

		Returns:
			All symbols valid for use in the expression.
		"""
		return [sym.sp_symbol_matsym for sym in self.all_sorted_symbols]

	####################
	# - JIT'ed Callables for Numerical Function Arguments
	####################
	@functools.cached_property
	def func_args_n(
		self,
	) -> list[
		typ.Callable[
			[int | float | complex | jtyp.Inexact[jtyp.Array, '...'], ...],
			int | float | complex | jtyp.Inexact[jtyp.Array, '...'],
		]
	]:
		"""Callable functions for evaluating each `self.func_args` entry numerically.

		Before simplification, each `self.func_args` entry will be conformed to the corresponding (by-index) `SimSymbol` in `self.target_syms`.

		Notes:
			Before using any `sympy` expressions as arguments to the returned callablees, they **must** be fully conformed and scaled to the corresponding `self.symbols` entry using that entry's `SimSymbol.scale()` method.

			This ensures conformance to the `SimSymbol` properties (like units), as well as adherance to a numerical type identity compatible with `sp.lambdify()`.
		"""
		return [
			sp.lambdify(
				self.all_sorted_sp_symbols,
				target_sym.conform(func_arg, strip_unit=True),
				'jax',
			)
			for func_arg, target_sym in zip(
				self.func_args, self.arg_targets, strict=True
			)
		]

	@functools.cached_property
	def func_kwargs_n(
		self,
	) -> dict[
		str,
		typ.Callable[
			[int | float | complex | jtyp.Inexact[jtyp.Array, '...'], ...],
			int | float | complex | jtyp.Inexact[jtyp.Array, '...'],
		],
	]:
		"""Callable functions for evaluating each `self.func_kwargs` entry numerically.

		The arguments of each function **must** be pre-treated using `SimSymbol.scale()`.
		This ensures conformance to the `SimSymbol` properties, as well as adherance to a numerical type identity compatible with `sp.lambdify()`
		"""
		return {
			key: sp.lambdify(
				self.all_sorted_sp_symbols,
				self.kwarg_targets[key].conform(func_arg, strip_unit=True),
				'jax',
			)
			for key, func_arg in self.func_kwargs.items()
		}

	####################
	# - Realization
	####################
	def realize_symbols(
		self,
		symbol_values: dict[
			sim_symbols.SimSymbol, spux.SympyExpr | RangeFlow | ArrayFlow
		] = MappingProxyType({}),
		allow_partial: bool = False,
	) -> dict[
		sim_symbols.SimSymbol,
		int | float | Fraction | float | complex | jtyp.Shaped[jtyp.Array, '...'] :,
	]:
		"""Fully realize all symbols by assigning them a value.

		Three kinds of values for `symbol_values` are supported, fundamentally:

		- **Sympy Expression**: When the value is a sympy expression with units, the unit of the `SimSymbol` key which unit the value if converted to.
			If the `SimSymbol`'s unit is `None`, then the value is left as-is.
		- **Range**: When the value is a `RangeFlow`, units are converted to the `SimSymbol`'s unit using `.rescale_to_unit()`.
			If the `SimSymbol`'s unit is `None`, then the value is left as-is.
		- **Array**: When the value is an `ArrayFlow`, units are converted to the `SimSymbol`'s unit using `.rescale_to_unit()`.
			If the `SimSymbol`'s unit is `None`, then the value is left as-is.

		Returns:
			A dictionary almost with `.subs()`, other than `jax` arrays.
		"""
		if allow_partial or set(self.all_sorted_symbols) == set(symbol_values.keys()):
			realized_syms = {}
			for sym in self.all_sorted_symbols:
				sym_value = symbol_values.get(sym)
				if sym_value is None and allow_partial:
					continue

				if isinstance(sym_value, spux.SympyType):
					v = sym.scale(sym_value)

				elif isinstance(sym_value, ArrayFlow | RangeFlow):
					v = sym_value.rescale_to_unit(sym.unit).values
					## NOTE: RangeFlow must not be symbolic.

				else:
					msg = f'No support for symbolic value {sym_value} (type={type(sym_value)})'
					raise NotImplementedError(msg)

				realized_syms |= {sym: v}

			return realized_syms

		msg = f'ParamsFlow: Not all symbols were given a value during realization (symbols={self.symbols}, symbol_values={symbol_values})'
		raise ValueError(msg)

	####################
	# - Realize Arguments
	####################
	def scaled_func_args(
		self,
		symbol_values: dict[sim_symbols.SimSymbol, spux.SympyExpr] = MappingProxyType(
			{}
		),
	) -> list[
		int | float | Fraction | float | complex | jtyp.Shaped[jtyp.Array, '...']
	]:
		"""Realize correctly conformed numerical arguments for `self.func_args`.

		Because we allow symbols to be used in `self.func_args`, producing a numerical value that can be passed directly to a `FuncFlow` becomes a two-step process:

		1. Conform Symbols: Arbitrary `sympy` expressions passed as `symbol_values` must first be conformed to match the ex. units of `SimSymbol`s found in `self.symbols`, before they can be used.

		2. Conform Function Arguments: Arbitrary `sympy` expressions encoded in `self.func_args` must, **after** inserting the conformed numerical symbols, themselves be conformed to the expected ex. units of the function that they are to be used within.

		Our implementation attempts to utilize simple, powerful primitives to accomplish this in roughly three steps:

		1. **Realize Symbols**: Particular passed symbolic values `symbol_values`, which are arbitrary `sympy` expressions, are conformed to the definitions in `self.symbols` (ex. to match units), then cast to numerical values (pure Python / jax array).

		2. **Lazy Function Arguments**: Stored function arguments `self.func_args`, which are arbitrary `sympy` expressions, are conformed to the definitions in `self.target_syms` (ex. to match units), then cast to numerical values (pure Python / jax array).
			_Technically, this happens as part of `self.func_args_n`._

		3. **Numerical Evaluation**: The numerical values for each symbol are passed as parameters to each (callable) element of `self.func_args_n`, which produces a correct numerical value for each function argument.

		Parameters:
			symbol_values: Particular values for all symbols in `self.symbols`, which will be conformed and used to compute the function arguments (before they are conformed to `self.target_syms`).
		"""
		realized_symbols = list(
			self.realize_symbols(symbol_values | self.realized_symbols).values()
		)
		return [func_arg_n(*realized_symbols) for func_arg_n in self.func_args_n]

	def scaled_func_kwargs(
		self,
		symbol_values: dict[spux.Symbol, spux.SympyExpr] = MappingProxyType({}),
	) -> dict[
		str, int | float | Fraction | float | complex | jtyp.Shaped[jtyp.Array, '...']
	]:
		"""Realize correctly conformed numerical arguments for `self.func_kwargs`.

		Other than the `dict[str, ...]` key, the semantics are identical to `self.scaled_func_args()`.
		"""
		realized_symbols = self.realize_symbols(symbol_values | self.realized_symbols)

		return {
			func_arg_name: func_kwarg_n(**realized_symbols)
			for func_arg_name, func_kwarg_n in self.func_kwargs_n.items()
		}

	####################
	# - Operations
	####################
	def __or__(
		self,
		other: typ.Self,
	):
		"""Combine two function parameter lists, such that the LHS will be concatenated with the RHS.

		Just like its neighbor in `Func`, this effectively combines two functions with unique parameters.
		The next composed function will receive a tuple of two arrays, instead of just one, allowing binary operations to occur.
		"""
		return ParamsFlow(
			arg_targets=self.arg_targets + other.arg_targets,
			kwarg_targets=self.kwarg_targets | other.kwarg_targets,
			func_args=self.func_args + other.func_args,
			func_kwargs=self.func_kwargs | other.func_kwargs,
			symbols=self.symbols | other.symbols,
			realized_symbols=self.realized_symbols | other.realized_symbols,
		)

	def compose_within(
		self,
		enclosing_arg_targets: list[sim_symbols.SimSymbol] = (),
		enclosing_kwarg_targets: list[sim_symbols.SimSymbol] = (),
		enclosing_func_args: list[spux.SympyExpr] = (),
		enclosing_func_kwargs: dict[str, spux.SympyExpr] = MappingProxyType({}),
		enclosing_symbols: frozenset[sim_symbols.SimSymbol] = frozenset(),
	) -> typ.Self:
		return ParamsFlow(
			arg_targets=self.arg_targets + list(enclosing_arg_targets),
			kwarg_targets=self.kwarg_targets | dict(enclosing_kwarg_targets),
			func_args=self.func_args + list(enclosing_func_args),
			func_kwargs=self.func_kwargs | dict(enclosing_func_kwargs),
			symbols=self.symbols | enclosing_symbols,
			realized_symbols=self.realized_symbols,
		)

	def realize_partial(
		self,
		symbol_values: dict[
			sim_symbols.SimSymbol, spux.SympyExpr | RangeFlow | ArrayFlow
		],
	) -> typ.Self:
		"""Provide a particular expression/range/array to realize some symbols.

		Essentially removes symbols from `self.symbols`, and adds the symbol w/value to `self.realized_symbols`.
		As a result, only the still-unrealized symbols need to be passed at the time of realization (using ex. `self.scaled_func_args()`).

		Parameters:
			symbol_values: The value to realize for each `SimSymbol`.
				**All keys** must be identically matched to a single element of `self.symbol`.
				Can be empty, in which case an identical new `ParamsFlow` will be returned.

		Raises:
			ValueError: If any symbol in `symbol_values`
		"""
		syms = set(symbol_values.keys())
		if syms.issubset(self.symbols) or not syms:
			return ParamsFlow(
				arg_targets=self.arg_targets,
				kwarg_targets=self.kwarg_targets,
				func_args=self.func_args,
				func_kwargs=self.func_kwargs,
				symbols=self.symbols - syms,
				realized_symbols=self.realized_symbols | symbol_values,
			)
		msg = f'ParamsFlow: Not all partially realized symbols are defined on the ParamsFlow (symbols={self.symbols}, symbol_values={symbol_values})'
		raise ValueError(msg)

	####################
	# - Generate ExprSocketDef
	####################
	@functools.cached_property
	def sym_expr_infos(self) -> dict[str, ExprInfo]:
		"""Generate keyword arguments for defining all `ExprSocket`s needed to realize all `self.symbols`.

		Many nodes need actual data, and as such, they require that the user select actual values for any symbols in the `ParamsFlow`.
		The best way to do this is to create one `ExprSocket` for each symbol that needs realizing.

		Notes:
			This method is created for the purpose of being able to make this exact call in an `events.on_value_changed` method:
			```
			self.loose_input_sockets = {
				sym_name: sockets.ExprSocketDef(**expr_info)
				for sym_name, expr_info in params.sym_expr_infos(info).items()
			}
			```

		The `ExprInfo`s can be directly defererenced `**expr_info`)
		"""
		for sym in self.sorted_symbols:
			if sym.rows > 3 or sym.cols > 1:
				msg = 'No support for >Vec3 / Matrix values in ExprInfo'
				raise NotImplementedError(msg)

		return {
			sym: {
				'default_steps': 25,
			}
			| sym.expr_info
			for sym in self.sorted_symbols
		}
