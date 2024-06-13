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
import functools
import typing as typ

import jax.lax as jlax
import jax.numpy as jnp

from blender_maxwell.utils import logger, sim_symbols

from .. import contracts as ct

log = logger.get(__name__)


class FilterOperation(enum.StrEnum):
	"""Valid operations for the `FilterMathNode`.

	Attributes:
		DimToVec: Shift last dimension to output.
		DimsToMat: Shift last 2 dimensions to output.
		PinLen1: Remove a len(1) dimension.
		Pin: Remove a len(n) dimension by selecting a particular index.
		Swap: Swap the positions of two dimensions.
		ZScore15: The Z-score threshold of values
		ZScore30: The peak-to-peak along an axis.
	"""

	# Slice
	Slice = enum.auto()
	SliceIdx = enum.auto()

	# Pin
	PinLen1 = enum.auto()
	Pin = enum.auto()
	PinIdx = enum.auto()

	# Swizzle
	Swap = enum.auto()

	# Axis Filter
	ZScore15 = enum.auto()
	ZScore30 = enum.auto()

	####################
	# - UI
	####################
	@staticmethod
	def to_name(value: typ.Self) -> str:
		"""A human-readable UI-oriented name for a physical type."""
		FO = FilterOperation
		return {
			# Slice
			FO.Slice: '≈a[v₁:v₂]',
			FO.SliceIdx: '=a[i:j]',
			# Pin
			FO.PinLen1: 'a[0] → a',
			FO.Pin: 'a[v] ⇝ a',
			FO.PinIdx: 'a[i] → a',
			# Swizzle
			FO.Swap: 'a₁ ↔ a₂',
			# Axis Filter
			# FO.ZScore15: 'a[v₁:v₂] ∈ σ[1.5]',
			# FO.ZScore30: 'a[v₁:v₂] ∈ σ[1.5]',
		}[value]

	@staticmethod
	def to_icon(_: typ.Self) -> str:
		"""No icons."""
		return ''

	def bl_enum_element(self, i: int) -> ct.BLEnumElement:
		"""Given an integer index, generate an element that conforms to the requirements of `bpy.props.EnumProperty.items`."""
		FO = FilterOperation
		return (
			str(self),
			FO.to_name(self),
			FO.to_name(self),
			FO.to_icon(self),
			i,
		)

	@staticmethod
	def bl_enum_elements(info: ct.InfoFlow) -> list[ct.BLEnumElement]:
		"""Generate a list of guaranteed-valid operations based on the passed `InfoFlow`s.

		Returns a `bpy.props.EnumProperty.items`-compatible list.
		"""
		return [
			operation.bl_enum_element(i)
			for i, operation in enumerate(FilterOperation.by_info(info))
		]

	####################
	# - Ops from Info
	####################
	@staticmethod
	def by_info(info: ct.InfoFlow) -> list[typ.Self]:
		FO = FilterOperation
		ops = []

		if info.dims:
			# Slice
			ops += [FO.SliceIdx]

			# Pin
			## PinLen1
			## -> There must be a dimension with length 1.
			if 1 in [
				len(dim_idx) for dim_idx in info.dims.values() if dim_idx is not None
			]:
				ops += [FO.PinLen1]

			# Pin
			## -> There must be a dimension, full stop.
			ops += [FO.Pin, FO.PinIdx]

			# Swizzle
			## Swap
			## -> There must be at least two dimensions to swap between.
			if len(info.dims) >= 2:  # noqa: PLR2004
				ops += [FO.Swap]

			# Axis Filter
			## ZScore
			## -> Subjectively, it makes little sense with less than 5 numbers.
			## -> Mathematically valid (I suppose) for 2. But not so useful.
			# if any(
			# (dim.has_idx_discrete(dim) or dim.has_idx_labels(dim))
			# and len(dim_idx) > 5  # noqa: PLR2004
			# for dim, dim_idx in info.dims.items()
			# ):
			# ops += [FO.ZScore15, FO.ZScore30]

		return ops

	####################
	# - Computed Properties
	####################
	@functools.cached_property
	def func_args(self) -> list[sim_symbols.SimSymbol]:
		FO = FilterOperation
		return {
			# Pin
			FO.Pin: [sim_symbols.idx(None)],
			FO.PinIdx: [sim_symbols.idx(None)],
			# Swizzle
			## -> Swap: JAX requires that swap dims be baked into the function.
			# Axis Filter
			# FO.ZScore15: [sim_symbols.idx(None)],
			# FO.ZScore30: [sim_symbols.idx(None)],
		}.get(self, [])

	####################
	# - Methods
	####################
	@functools.cached_property
	def num_dim_inputs(self) -> None:
		"""Number of dimensions required as inputs to the operation's function."""
		FO = FilterOperation
		return {
			# Slice
			FO.Slice: 1,
			FO.SliceIdx: 1,
			# Pin
			FO.PinLen1: 1,
			FO.Pin: 1,
			FO.PinIdx: 1,
			# Swizzle
			FO.Swap: 2,
			# Axis Filter
			# FO.ZScore15: 1,
			# FO.ZScore30: 1,
		}[self]

	def valid_dims(self, info: ct.InfoFlow) -> list[typ.Self]:
		"""The valid dimensions that can be selected between, fo each of the"""
		FO = FilterOperation
		match self:
			# Slice
			case FO.Slice:
				return [dim for dim in info.dims if not info.has_idx_labels(dim)]

			case FO.SliceIdx:
				return [dim for dim in info.dims if not info.has_idx_labels(dim)]

			# Pin
			case FO.PinLen1:
				return [
					dim
					for dim, dim_idx in info.dims.items()
					if not info.has_idx_cont(dim) and len(dim_idx) == 1
				]

			case FO.Pin:
				return info.dims

			case FO.PinIdx:
				return [dim for dim in info.dims if not info.has_idx_cont(dim)]

			# Dimension
			case FO.Swap:
				return info.dims

			# TODO: ZScore

		return []

	####################
	# - Implementations
	####################
	def jax_func(
		self,
		axis_0: int | None,
		axis_1: int | None = None,
		slice_tuple: tuple[int, int, int] | None = None,
	):
		"""Implements the identified filtering using `jax`."""
		FO = FilterOperation
		return {
			# Pin
			FO.Slice: lambda expr: jlax.slice_in_dim(
				expr, slice_tuple[0], slice_tuple[1], slice_tuple[2], axis=axis_0
			),
			FO.SliceIdx: lambda expr: jlax.slice_in_dim(
				expr, slice_tuple[0], slice_tuple[1], slice_tuple[2], axis=axis_0
			),
			# Pin
			FO.PinLen1: lambda expr: jnp.squeeze(expr, axis_0),
			FO.Pin: lambda expr, idx: jnp.take(expr, idx, axis=axis_0),
			FO.PinIdx: lambda expr, idx: jnp.take(expr, idx, axis=axis_0),
			# Dimension
			FO.Swap: lambda expr: jnp.swapaxes(expr, axis_0, axis_1),
			# TODO: Axis Filters
			## -> The jnp.compress() function is ideal for this kind of thing.
			## -> The difficulty is that jit() requires output size to be known.
			## -> One can set the size= parameter of compress.
			## -> But how do we determine that?
		}[self]

	####################
	# - Transforms
	####################
	def transform_func(
		self,
		func: ct.FuncFlow,
		axis_0: int,
		axis_1: int | None = None,
		slice_tuple: tuple[int, int, int] | None = None,
	) -> ct.FuncFlow | None:
		"""Transform input function according to the current operation and output info characterization."""
		FO = FilterOperation
		match self:
			# Slice
			case FO.Slice | FO.SliceIdx if axis_0 is not None:
				return func.compose_within(
					self.jax_func(axis_0, slice_tuple=slice_tuple),
					enclosing_func_output=func.func_output,
					supports_jax=True,
				)

			# Pin
			case FO.PinLen1 if axis_0 is not None:
				return func.compose_within(
					self.jax_func(axis_0),
					enclosing_func_output=func.func_output,
					supports_jax=True,
				)

			case FO.Pin | FO.PinIdx if axis_0 is not None:
				return func.compose_within(
					self.jax_func(axis_0),
					enclosing_func_args=[sim_symbols.idx(None)],
					enclosing_func_output=func.func_output,
					supports_jax=True,
				)

			# Swizzle
			case FO.Swap if axis_0 is not None and axis_1 is not None:
				return func.compose_within(
					self.jax_func(axis_0, axis_1),
					enclosing_func_output=func.func_output,
					supports_jax=True,
				)

		return None

	def transform_info(
		self,
		info: ct.InfoFlow,
		dim_0: sim_symbols.SimSymbol,
		dim_1: sim_symbols.SimSymbol | None = None,
		pin_idx: int | None = None,
		slice_tuple: tuple[int, int, int] | None = None,
	):
		FO = FilterOperation
		return {
			FO.Slice: lambda: info.slice_dim(dim_0, slice_tuple),
			FO.SliceIdx: lambda: info.slice_dim(dim_0, slice_tuple),
			# Pin
			FO.PinLen1: lambda: info.delete_dim(dim_0, pin_idx=0),
			FO.Pin: lambda: info.delete_dim(dim_0, pin_idx=pin_idx),
			FO.PinIdx: lambda: info.delete_dim(dim_0, pin_idx=pin_idx),
			# Reinterpret
			FO.Swap: lambda: info.swap_dimensions(dim_0, dim_1),
			# TODO: Axis Filters
		}[self]()
