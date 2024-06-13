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

import enum
import typing as typ

import jax.numpy as jnp
import jaxtyping as jtyp
import sympy as sp

from blender_maxwell.utils import logger, sim_symbols
from blender_maxwell.utils import sympy_extra as spux

from .. import contracts as ct

log = logger.get(__name__)

MT = spux.MathType
PT = spux.PhysicalType


class ReduceOperation(enum.StrEnum):
	"""Valid operations for the `ReduceMathNode`.

	Attributes:
		Count: The number of discrete elements along an axis.
		Mean: The average along an axis.
		Std: The standard deviation along an axis.
		Var: The variance along an axis.
		Min: The minimum value along an axis.
		Q25: The `25%` quantile along an axis.
		Medium: The `50%` quantile along an axis.
		Q75: The `75%` quantile along an axis.
		Max: The `75%` quantile along an axis.
		P2P: The peak-to-peak range along an axis.
		Z15ToZ15: The range between z-scores of 1.5 in each direction.
		Z30ToZ30: The range between z-scores of 3.0 in each direction.
		Sum: The sum along an axis.
		Prod: The product along an axis.
	"""

	# Summary
	Count = enum.auto()

	# Statistics
	Mean = enum.auto()
	Std = enum.auto()
	Var = enum.auto()

	Min = enum.auto()
	Q25 = enum.auto()
	Median = enum.auto()
	Q75 = enum.auto()
	Max = enum.auto()

	P2P = enum.auto()
	Z15ToZ15 = enum.auto()
	Z30ToZ30 = enum.auto()

	# Reductions
	Sum = enum.auto()
	Prod = enum.auto()

	####################
	# - UI
	####################
	@staticmethod
	def to_name(value: typ.Self) -> str:
		"""A human-readable UI-oriented name for a physical type."""
		RO = ReduceOperation
		return {
			# Summary
			RO.Count: '# [a]',
			# Statistics
			RO.Mean: 'μ [a]',
			RO.Std: 'σ [a]',
			RO.Var: 'σ² [a]',
			RO.Min: 'min [a]',
			RO.Q25: 'q₂₅ [a]',
			RO.Median: 'median [a]',
			RO.Q75: 'q₇₅ [a]',
			RO.Max: 'max [a]',
			RO.P2P: 'p2p [a]',
			RO.Z15ToZ15: 'σ[1.5] [a]',
			RO.Z30ToZ30: 'σ[3.0] [a]',
			# Reductions
			RO.Sum: 'sum [a]',
			RO.Prod: 'prod [a]',
		}[value]

	@property
	def name(self) -> str:
		return ReduceOperation.to_name(self)

	@staticmethod
	def to_icon(_: typ.Self) -> str:
		"""No icons."""
		return ''

	def bl_enum_element(self, i: int) -> ct.BLEnumElement:
		"""Given an integer index, generate an element that conforms to the requirements of `bpy.props.EnumProperty.items`."""
		RO = ReduceOperation
		return (
			str(self),
			RO.to_name(self),
			RO.to_name(self),
			RO.to_icon(self),
			i,
		)

	@staticmethod
	def bl_enum_elements(info: ct.InfoFlow) -> list[ct.BLEnumElement]:
		"""Generate a list of guaranteed-valid operations based on the passed `InfoFlow`s.

		Returns a `bpy.props.EnumProperty.items`-compatible list.
		"""
		return [
			operation.bl_enum_element(i)
			for i, operation in enumerate(ReduceOperation.from_info(info))
		]

	####################
	# - Derivation
	####################
	@staticmethod
	def from_info(info: ct.InfoFlow) -> list[typ.Self]:
		"""Derive valid reduction operations from the `InfoFlow` of the operand."""
		RO = ReduceOperation
		ops = []

		if info.dims and any(
			info.has_idx_discrete(dim) or info.has_idx_labels(dim) for dim in info.dims
		):
			# Summary
			ops += [RO.Count]

			# Statistics
			ops += [
				RO.Mean,
				RO.Std,
				RO.Var,
				RO.Min,
				RO.Q25,
				RO.Median,
				RO.Q75,
				RO.Max,
				RO.P2P,
				RO.Z15ToZ15,
				RO.Z30ToZ30,
			]

			# Reductions
			ops += [RO.Sum, RO.Prod]

			## I know, they can be combined.
			## But they may one day need more checks.

		return ops

	def valid_dims(self, info: ct.InfoFlow) -> list[typ.Self]:
		"""Valid dimensions that can be reduced."""
		return [
			dim
			for dim in info.dims
			if info.has_idx_discrete(dim) or info.has_idx_labels(dim)
		]

	####################
	# - Composable Functions
	####################
	@property
	def jax_func(
		self,
	) -> typ.Callable[
		[jtyp.Shaped[jtyp.Array, '...'], int], jtyp.Shaped[jtyp.Array, '...']
	]:
		"""Implements the identified reduction using `jax`."""
		RO = ReduceOperation
		return {
			# Summary
			RO.Count: lambda el, axis: el.shape[axis],
			# RO.Mode: lambda el, axis: jsc.stats.mode(el, axis=axis).
			# Statistics
			RO.Mean: lambda el, axis: jnp.mean(el, axis=axis),
			RO.Std: lambda el, axis: jnp.std(el, axis=axis),
			RO.Var: lambda el, axis: jnp.var(el, axis=axis),
			RO.Min: lambda el, axis: jnp.min(el, axis=axis),
			RO.Q25: lambda el, axis: jnp.quantile(el, 0.25, axis=axis),
			RO.Median: lambda el, axis: jnp.median(el, axis=axis),
			RO.Q75: lambda el, axis: jnp.quantile(el, 0.75, axis=axis),
			RO.Max: lambda el, axis: jnp.max(el, axis=axis),
			RO.P2P: lambda el, axis: jnp.ptp(el, axis=axis),
			RO.Z15ToZ15: lambda el, axis: 2 * (3 / 2) * jnp.std(el, axis=axis),
			RO.Z30ToZ30: lambda el, axis: 2 * 3 * jnp.std(el, axis=axis),
			# Statistics
			RO.Sum: lambda el, axis: jnp.sum(el, axis=axis),
			RO.Prod: lambda el, axis: jnp.prod(el, axis=axis),
		}[self]

	####################
	# - Transforms
	####################
	def transform_func(self, func: ct.InfoFlow):
		"""Transform the lazy `FuncFlow` to reduce the input."""
		return func.compose_within(
			self.jax_func,
			enclosing_func_args=(sim_symbols.idx(None),),
			enclosing_func_output=self.transform_output(func.func_output),
			supports_jax=True,
		)

	def transform_info(self, info: ct.InfoFlow, dim: sim_symbols.SimSymbol):
		"""Transform the characterizing `InfoFlow` of the reduced operand."""
		return info.delete_dim(dim).update(output=self.transform_output(info.output))

	def transform_params(self, params: ct.ParamsFlow, axis: int) -> None:
		"""Transform the characterizing `InfoFlow` of the reduced operand."""
		return params.compose_within(
			enclosing_func_args=(sp.Integer(axis),),
		)

	def transform_output(self, sym: sim_symbols.SimSymbol) -> sim_symbols.SimSymbol:
		"""Transform the domain of the output symbol.

		Parameters:
			dom: Symbolic set representing the original output symbol's domain.
			info: Characterization of the original expression.
			dim: Dimension symbol being reduced away.

		"""
		RO = ReduceOperation

		match self:
			# Summary
			case RO.Count:
				return sym.update(
					sym_name=sim_symbols.SimSymbolName.Count,
					mathtype=MT.Integer,
					physical_type=PT.NonPhysical,
					unit=None,
					rows=1,
					cols=1,
					domain=spux.BlessedSet(sp.Naturals0),
				)

			# Statistics
			case (
				RO.Mean
				| RO.Std
				| RO.Var
				| RO.Min
				| RO.Q25
				| RO.Median
				| RO.Q75
				| RO.Max
				| RO.P2P
				| RO.Z15ToZ15
				| RO.Z30ToZ30
			):
				## -> Stats are enclosed by the original domain.
				return sym

			# Reductions
			case RO.Sum:
				return sym.update(
					domain=spux.BlessedSet(sym.mathtype.symbolic_set),
				)

			case RO.Prod:
				return sym.update(
					domain=spux.BlessedSet(sym.mathtype.symbolic_set),
				)
