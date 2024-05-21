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

from .flow_kinds import FlowKind
from .info import InfoFlow

log = logger.get(__name__)


class ExprInfo(typ.TypedDict):
	active_kind: FlowKind
	size: spux.NumberSize1D
	mathtype: spux.MathType
	physical_type: spux.PhysicalType

	# Value
	default_value: spux.SympyExpr

	# Range
	default_min: spux.SympyExpr
	default_max: spux.SympyExpr
	default_steps: int


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
	# - Realize Arguments
	####################
	def scaled_func_args(
		self,
		unit_system: spux.UnitSystem,
		symbol_values: dict[spux.Symbol, spux.SympyExpr] = MappingProxyType({}),
	):
		"""Realize the function arguments contained in this `ParamsFlow`, making it ready for insertion into `Func.func()`.

		For all `arg`s in `self.func_args`, the following operations are performed:
		- **Unit System**: If `arg`


		Notes:
			This method is created for the purpose of being able to make this exact call in an `events.on_value_changed` method:

		"""

		"""Return the function arguments, scaled to the unit system, stripped of units, and cast to jax-compatible arguments."""
		if not all(sym in self.symbols for sym in symbol_values):
			msg = f"Symbols in {symbol_values} don't perfectly match the ParamsFlow symbols {self.symbols}"
			raise ValueError(msg)

		## TODO: MutableDenseMatrix causes error with 'in' check bc it isn't hashable.
		return [
			(
				spux.scale_to_unit_system(arg, unit_system, use_jax_array=True)
				if arg not in symbol_values
				else symbol_values[arg]
			)
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

		Just like its neighbor in `Func`, this effectively combines two functions with unique parameters.
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

	####################
	# - Generate ExprSocketDef
	####################
	def sym_expr_infos(
		self, info: InfoFlow, use_range: bool = False
	) -> dict[str, ExprInfo]:
		"""Generate all information needed to define expressions that realize all symbolic parameters in this `ParamsFlow`.

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

		Parameters:
			info: The InfoFlow associated with the `Expr` being realized.
				Each symbol in `self.symbols` **must** have an associated same-named dimension in `info`.
			use_range: Causes the

		The `ExprInfo`s can be directly defererenced `**expr_info`)
		"""
		return {
			sym.name: {
				# Declare Kind/Size
				## -> Kind: Value prevents user-alteration of config.
				## -> Size: Always scalar, since symbols are scalar (for now).
				'active_kind': FlowKind.Value,
				'size': spux.NumberSize1D.Scalar,
				# Declare MathType/PhysicalType
				## -> MathType: Lookup symbol name in info dimensions.
				## -> PhysicalType: Same.
				'mathtype': info.dim_mathtypes[sym.name],
				'physical_type': info.dim_physical_types[sym.name],
				# TODO: Default Values
				# FlowKind.Value: Default Value
				#'default_value':
				# FlowKind.Range: Default Min/Max/Steps
				#'default_min':
				#'default_max':
				#'default_steps':
			}
			for sym in self.sorted_symbols
			if sym.name in info.dim_names
		}
