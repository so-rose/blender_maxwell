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
import functools
import typing as typ
from types import MappingProxyType

import sympy as sp

from blender_maxwell.utils import extra_sympy_units as spux
from blender_maxwell.utils import logger

log = logger.get(__name__)


@dataclasses.dataclass(frozen=True, kw_only=True)
class ParamsFlow:
	func_args: list[spux.SympyExpr] = dataclasses.field(default_factory=list)
	func_kwargs: dict[str, spux.SympyExpr] = dataclasses.field(default_factory=dict)

	symbols: frozenset[spux.Symbol] = frozenset()

	@functools.cached_property
	def sorted_symbols(self) -> list[sp.Symbol]:
		"""Retrieves all symbols by concatenating int, real, and complex symbols, and sorting them by name.

		Returns:
			All symbols valid for use in the expression.
		"""
		return sorted(self.symbols, key=lambda sym: sym.name)

	####################
	# - Scaled Func Args
	####################
	def scaled_func_args(
		self,
		unit_system: spux.UnitSystem,
		symbol_values: dict[spux.Symbol, spux.SympyExpr] = MappingProxyType({}),
	):
		"""Return the function arguments, scaled to the unit system, stripped of units, and cast to jax-compatible arguments."""
		if not all(sym in self.symbols for sym in symbol_values):
			msg = f"Symbols in {symbol_values} don't perfectly match the ParamsFlow symbols {self.symbols}"
			raise ValueError(msg)

		## TODO: MutableDenseMatrix causes error with 'in' check bc it isn't hashable.
		return [
			spux.scale_to_unit_system(arg, unit_system, use_jax_array=True)
			if arg not in symbol_values
			else symbol_values[arg]
			for arg in self.func_args
		]

	def scaled_func_kwargs(
		self,
		unit_system: spux.UnitSystem,
		symbol_values: dict[spux.Symbol, spux.SympyExpr] = MappingProxyType({}),
	):
		"""Return the function arguments, scaled to the unit system, stripped of units, and cast to jax-compatible arguments."""
		if not all(sym in self.symbols for sym in symbol_values):
			msg = f"Symbols in {symbol_values} don't perfectly match the ParamsFlow symbols {self.symbols}"
			raise ValueError(msg)

		return {
			arg_name: spux.convert_to_unit_system(arg, unit_system, use_jax_array=True)
			if arg not in symbol_values
			else symbol_values[arg]
			for arg_name, arg in self.func_kwargs.items()
		}

	####################
	# - Operations
	####################
	def __or__(
		self,
		other: typ.Self,
	):
		"""Combine two function parameter lists, such that the LHS will be concatenated with the RHS.

		Just like its neighbor in `LazyValueFunc`, this effectively combines two functions with unique parameters.
		The next composed function will receive a tuple of two arrays, instead of just one, allowing binary operations to occur.
		"""
		return ParamsFlow(
			func_args=self.func_args + other.func_args,
			func_kwargs=self.func_kwargs | other.func_kwargs,
			symbols=self.symbols | other.symbols,
		)

	def compose_within(
		self,
		enclosing_func_args: list[spux.SympyExpr] = (),
		enclosing_func_kwargs: dict[str, spux.SympyExpr] = MappingProxyType({}),
		enclosing_symbols: frozenset[spux.Symbol] = frozenset(),
	) -> typ.Self:
		return ParamsFlow(
			func_args=self.func_args + list(enclosing_func_args),
			func_kwargs=self.func_kwargs | dict(enclosing_func_kwargs),
			symbols=self.symbols | enclosing_symbols,
		)
