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

import jax

from blender_maxwell.utils import extra_sympy_units as spux
from blender_maxwell.utils import logger

from .array import ArrayFlow
from .lazy_range import RangeFlow

log = logger.get(__name__)


@dataclasses.dataclass(frozen=True, kw_only=True)
class InfoFlow:
	####################
	# - Covariant Input
	####################
	dim_names: list[str] = dataclasses.field(default_factory=list)
	dim_idx: dict[str, ArrayFlow | RangeFlow] = dataclasses.field(
		default_factory=dict
	)  ## TODO: Rename to dim_idxs

	@functools.cached_property
	def dim_has_coords(self) -> dict[str, int]:
		return {
			dim_name: not (
				isinstance(dim_idx, RangeFlow)
				and (dim_idx.start.is_infinite or dim_idx.stop.is_infinite)
			)
			for dim_name, dim_idx in self.dim_idx.items()
		}

	@functools.cached_property
	def dim_lens(self) -> dict[str, int]:
		return {dim_name: len(dim_idx) for dim_name, dim_idx in self.dim_idx.items()}

	@functools.cached_property
	def dim_mathtypes(self) -> dict[str, spux.MathType]:
		return {
			dim_name: dim_idx.mathtype for dim_name, dim_idx in self.dim_idx.items()
		}

	@functools.cached_property
	def dim_units(self) -> dict[str, spux.Unit]:
		return {dim_name: dim_idx.unit for dim_name, dim_idx in self.dim_idx.items()}

	@functools.cached_property
	def dim_physical_types(self) -> dict[str, spux.PhysicalType]:
		return {
			dim_name: spux.PhysicalType.from_unit(dim_idx.unit)
			for dim_name, dim_idx in self.dim_idx.items()
		}

	@functools.cached_property
	def dim_idx_arrays(self) -> list[jax.Array]:
		return [
			dim_idx.realize().values
			if isinstance(dim_idx, RangeFlow)
			else dim_idx.values
			for dim_idx in self.dim_idx.values()
		]

	####################
	# - Contravariant Output
	####################
	# Output Information
	## TODO: Add PhysicalType
	output_name: str = dataclasses.field(default_factory=list)
	output_shape: tuple[int, ...] | None = dataclasses.field(default=None)
	output_mathtype: spux.MathType = dataclasses.field()
	output_unit: spux.Unit | None = dataclasses.field()

	@property
	def output_shape_len(self) -> int:
		if self.output_shape is None:
			return 0
		return len(self.output_shape)

	# Pinned Dimension Information
	## TODO: Add PhysicalType
	pinned_dim_names: list[str] = dataclasses.field(default_factory=list)
	pinned_dim_values: dict[str, float | complex] = dataclasses.field(
		default_factory=dict
	)
	pinned_dim_mathtypes: dict[str, spux.MathType] = dataclasses.field(
		default_factory=dict
	)
	pinned_dim_units: dict[str, spux.Unit] = dataclasses.field(default_factory=dict)

	####################
	# - Methods
	####################
	def slice_dim(self, dim_name: str, slice_tuple: tuple[int, int, int]) -> typ.Self:
		return InfoFlow(
			# Dimensions
			dim_names=self.dim_names,
			dim_idx={
				_dim_name: (
					dim_idx
					if _dim_name != dim_name
					else dim_idx[slice_tuple[0] : slice_tuple[1] : slice_tuple[2]]
				)
				for _dim_name, dim_idx in self.dim_idx.items()
			},
			# Outputs
			output_name=self.output_name,
			output_shape=self.output_shape,
			output_mathtype=self.output_mathtype,
			output_unit=self.output_unit,
		)

	def replace_dim(
		self, old_dim_name: str, new_dim_idx: tuple[str, ArrayFlow | RangeFlow]
	) -> typ.Self:
		"""Replace a dimension (and its indexing) with a new name and index array/range."""
		return InfoFlow(
			# Dimensions
			dim_names=[
				dim_name if dim_name != old_dim_name else new_dim_idx[0]
				for dim_name in self.dim_names
			],
			dim_idx={
				(dim_name if dim_name != old_dim_name else new_dim_idx[0]): (
					dim_idx if dim_name != old_dim_name else new_dim_idx[1]
				)
				for dim_name, dim_idx in self.dim_idx.items()
			},
			# Outputs
			output_name=self.output_name,
			output_shape=self.output_shape,
			output_mathtype=self.output_mathtype,
			output_unit=self.output_unit,
		)

	def rescale_dim_idxs(self, new_dim_idxs: dict[str, RangeFlow]) -> typ.Self:
		"""Replace several dimensional indices with new index arrays/ranges."""
		return InfoFlow(
			# Dimensions
			dim_names=self.dim_names,
			dim_idx={
				_dim_name: new_dim_idxs.get(_dim_name, dim_idx)
				for _dim_name, dim_idx in self.dim_idx.items()
			},
			# Outputs
			output_name=self.output_name,
			output_shape=self.output_shape,
			output_mathtype=self.output_mathtype,
			output_unit=self.output_unit,
		)

	def delete_dimension(self, dim_name: str) -> typ.Self:
		"""Delete a dimension."""
		return InfoFlow(
			# Dimensions
			dim_names=[
				_dim_name for _dim_name in self.dim_names if _dim_name != dim_name
			],
			dim_idx={
				_dim_name: dim_idx
				for _dim_name, dim_idx in self.dim_idx.items()
				if _dim_name != dim_name
			},
			# Outputs
			output_name=self.output_name,
			output_shape=self.output_shape,
			output_mathtype=self.output_mathtype,
			output_unit=self.output_unit,
		)

	def swap_dimensions(self, dim_0_name: str, dim_1_name: str) -> typ.Self:
		"""Swap the position of two dimensions."""

		# Compute Swapped Dimension Name List
		def name_swapper(dim_name):
			return (
				dim_name
				if dim_name not in [dim_0_name, dim_1_name]
				else {dim_0_name: dim_1_name, dim_1_name: dim_0_name}[dim_name]
			)

		dim_names = [name_swapper(dim_name) for dim_name in self.dim_names]

		# Compute Info
		return InfoFlow(
			# Dimensions
			dim_names=dim_names,
			dim_idx={dim_name: self.dim_idx[dim_name] for dim_name in dim_names},
			# Outputs
			output_name=self.output_name,
			output_shape=self.output_shape,
			output_mathtype=self.output_mathtype,
			output_unit=self.output_unit,
		)

	def set_output_mathtype(self, output_mathtype: spux.MathType) -> typ.Self:
		"""Set the MathType of the output."""
		return InfoFlow(
			dim_names=self.dim_names,
			dim_idx=self.dim_idx,
			# Outputs
			output_name=self.output_name,
			output_shape=self.output_shape,
			output_mathtype=output_mathtype,
			output_unit=self.output_unit,
		)

	def collapse_output(
		self,
		collapsed_name: str,
		collapsed_mathtype: spux.MathType,
		collapsed_unit: spux.Unit,
	) -> typ.Self:
		"""Replace the (scalar) output with the given corrected values."""
		return InfoFlow(
			# Dimensions
			dim_names=self.dim_names,
			dim_idx=self.dim_idx,
			output_name=collapsed_name,
			output_shape=None,
			output_mathtype=collapsed_mathtype,
			output_unit=collapsed_unit,
		)

	@functools.cached_property
	def shift_last_input(self):
		"""Shift the last input dimension to the output."""
		return InfoFlow(
			# Dimensions
			dim_names=self.dim_names[:-1],
			dim_idx={
				dim_name: dim_idx
				for dim_name, dim_idx in self.dim_idx.items()
				if dim_name != self.dim_names[-1]
			},
			# Outputs
			output_name=self.output_name,
			output_shape=(
				(self.dim_lens[self.dim_names[-1]],)
				if self.output_shape is None
				else (self.dim_lens[self.dim_names[-1]], *self.output_shape)
			),
			output_mathtype=self.output_mathtype,
			output_unit=self.output_unit,
		)
