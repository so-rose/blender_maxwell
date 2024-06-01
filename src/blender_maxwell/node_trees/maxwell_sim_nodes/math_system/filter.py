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
	"""

	# Slice
	Slice = enum.auto()
	SliceIdx = enum.auto()

	# Pin
	PinLen1 = enum.auto()
	Pin = enum.auto()
	PinIdx = enum.auto()

	# Dimension
	Swap = enum.auto()

	####################
	# - UI
	####################
	@staticmethod
	def to_name(value: typ.Self) -> str:
		FO = FilterOperation
		return {
			# Slice
			FO.Slice: '≈a[v₁:v₂]',
			FO.SliceIdx: '=a[i:j]',
			# Pin
			FO.PinLen1: 'a[0] → a',
			FO.Pin: 'a[v] ⇝ a',
			FO.PinIdx: 'a[i] → a',
			# Reinterpret
			FO.Swap: 'a₁ ↔ a₂',
		}[value]

	@staticmethod
	def to_icon(value: typ.Self) -> str:
		return ''

	def bl_enum_element(self, i: int) -> ct.BLEnumElement:
		FO = FilterOperation
		return (
			str(self),
			FO.to_name(self),
			FO.to_name(self),
			FO.to_icon(self),
			i,
		)

	####################
	# - Ops from Info
	####################
	@staticmethod
	def by_info(info: ct.InfoFlow) -> list[typ.Self]:
		FO = FilterOperation
		operations = []

		# Slice
		if info.dims:
			operations.append(FO.SliceIdx)

		# Pin
		## PinLen1
		## -> There must be a dimension with length 1.
		if 1 in [dim_idx for dim_idx in info.dims.values() if dim_idx is not None]:
			operations.append(FO.PinLen1)

		## Pin | PinIdx
		## -> There must be a dimension, full stop.
		if info.dims:
			operations += [FO.Pin, FO.PinIdx]

		# Reinterpret
		## Swap
		## -> There must be at least two dimensions.
		if len(info.dims) >= 2:  # noqa: PLR2004
			operations.append(FO.Swap)

		return operations

	####################
	# - Computed Properties
	####################
	@property
	def func_args(self) -> list[sim_symbols.SimSymbol]:
		FO = FilterOperation
		return {
			# Pin
			FO.Pin: [sim_symbols.idx(None)],
			FO.PinIdx: [sim_symbols.idx(None)],
		}.get(self, [])

	####################
	# - Methods
	####################
	@property
	def num_dim_inputs(self) -> None:
		FO = FilterOperation
		return {
			# Slice
			FO.Slice: 1,
			FO.SliceIdx: 1,
			# Pin
			FO.PinLen1: 1,
			FO.Pin: 1,
			FO.PinIdx: 1,
			# Reinterpret
			FO.Swap: 2,
		}[self]

	def valid_dims(self, info: ct.InfoFlow) -> list[typ.Self]:
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

		return []

	def are_dims_valid(
		self, info: ct.InfoFlow, dim_0: str | None, dim_1: str | None
	) -> bool:
		"""Check whether the given dimension inputs are valid in the context of this operation, and of the information."""
		if self.num_dim_inputs == 1:
			return dim_0 in self.valid_dims(info)

		if self.num_dim_inputs == 2:  # noqa: PLR2004
			valid_dims = self.valid_dims(info)
			return dim_0 in valid_dims and dim_1 in valid_dims

		return False

	####################
	# - UI
	####################
	def jax_func(
		self,
		axis_0: int | None,
		axis_1: int | None,
		slice_tuple: tuple[int, int, int] | None = None,
	):
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
		}[self]

	def transform_info(
		self,
		info: ct.InfoFlow,
		dim_0: sim_symbols.SimSymbol,
		dim_1: sim_symbols.SimSymbol,
		pin_idx: int | None = None,
		slice_tuple: tuple[int, int, int] | None = None,
	):
		FO = FilterOperation
		return {
			FO.Slice: lambda: info.slice_dim(dim_0, slice_tuple),
			FO.SliceIdx: lambda: info.slice_dim(dim_0, slice_tuple),
			# Pin
			FO.PinLen1: lambda: info.delete_dim(dim_0, pin_idx=pin_idx),
			FO.Pin: lambda: info.delete_dim(dim_0, pin_idx=pin_idx),
			FO.PinIdx: lambda: info.delete_dim(dim_0, pin_idx=pin_idx),
			# Reinterpret
			FO.Swap: lambda: info.swap_dimensions(dim_0, dim_1),
		}[self]()
