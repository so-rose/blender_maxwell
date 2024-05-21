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
from blender_maxwell.utils import logger, sim_symbols

from .expr_info import ExprInfo
from .flow_kinds import FlowKind

# from .info import InfoFlow

log = logger.get(__name__)


@dataclasses.dataclass(frozen=True, kw_only=True)
class ParamsFlow:
	func_args: list[spux.SympyExpr] = dataclasses.field(default_factory=list)
	func_kwargs: dict[str, spux.SympyExpr] = dataclasses.field(default_factory=dict)

	symbols: frozenset[sim_symbols.SimSymbol] = frozenset()

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
		unit_system: spux.UnitSystem | None = None,
		symbol_values: dict[sim_symbols.SimSymbol, spux.SympyExpr] = MappingProxyType(
			{}
		),
	):
		"""Realize the function arguments contained in this `ParamsFlow`, making it ready for insertion into `Func.func()`.

		For all `arg`s in `self.func_args`, the following operations are performed.

		Notes:
			This method is created for the purpose of being able to make this exact call in an `events.on_value_changed` method:
		"""
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
		unit_system: spux.UnitSystem | None = None,
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
	def sym_expr_infos(self, info, use_range: bool = False) -> dict[str, ExprInfo]:
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
		for sim_sym in self.sorted_symbols:
			if use_range and sim_sym.mathtype is spux.MathType.Complex:
				msg = 'No support for complex range in ExprInfo'
				raise NotImplementedError(msg)
			if use_range and (sim_sym.rows > 1 or sim_sym.cols > 1):
				msg = 'No support for non-scalar elements of range in ExprInfo'
				raise NotImplementedError(msg)
			if sim_sym.rows > 3 or sim_sym.cols > 1:
				msg = 'No support for >Vec3 / Matrix values in ExprInfo'
				raise NotImplementedError(msg)
		return {
			sim_sym.name: {
				# Declare Kind/Size
				## -> Kind: Value prevents user-alteration of config.
				## -> Size: Always scalar, since symbols are scalar (for now).
				'active_kind': FlowKind.Value if not use_range else FlowKind.Range,
				'size': spux.NumberSize1D.Scalar,
				# Declare MathType/PhysicalType
				## -> MathType: Lookup symbol name in info dimensions.
				## -> PhysicalType: Same.
				'mathtype': self.dims[sim_sym].mathtype,
				'physical_type': self.dims[sim_sym].physical_type,
				# TODO: Default Value
				# FlowKind.Value: Default Value
				#'default_value':
				# FlowKind.Range: Default Min/Max/Steps
				'default_min': sim_sym.domain.start,
				'default_max': sim_sym.domain.end,
				'default_steps': 50,
			}
			for sim_sym in self.sorted_symbols
		}
