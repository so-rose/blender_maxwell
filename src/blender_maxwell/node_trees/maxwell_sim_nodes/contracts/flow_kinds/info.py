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

from blender_maxwell.utils import extra_sympy_units as spux
from blender_maxwell.utils import logger, sim_symbols

from .array import ArrayFlow
from .lazy_range import RangeFlow

log = logger.get(__name__)

LabelArray: typ.TypeAlias = list[str]

# IndexArray: Identifies Discrete Dimension Values
## -> ArrayFlow (rat|real): Index by particular, not-guaranteed-uniform index.
## -> RangeFlow (rat|real): Index by unrealized array scaled between boundaries.
## -> LabelArray (int): For int-index arrays, interpret using these labels.
## -> None: Non-Discrete/unrealized indexing; use 'dim.domain'.
IndexArray: typ.TypeAlias = ArrayFlow | RangeFlow | LabelArray | None


@dataclasses.dataclass(frozen=True, kw_only=True)
class InfoFlow:
	"""Contains dimension and output information characterizing the array produced by a parallel `FuncFlow`.

	Functionally speaking, `InfoFlow` provides essential mathematical and physical context to raw array data, with terminology adapted from multilinear algebra.

	# From Arrays to Tensors
	The best way to illustrate how it works is to specify how raw-array concepts map to an array described by an `InfoFlow`:

	- **Index**: In raw arrays, the "index" is generally constrained to an integer ring, and has no semantic meaning.
		**(Covariant) Dimension**: The "dimension" is an named "index array", which assigns each integer index a **scalar** value of particular mathematical type, name, and unit (if not unitless).
	- **Value**: In raw arrays, the "value" is some particular computational type, or another raw array.
		**(Contravariant) Output**: The "output" is a strictly named, sized object that can only be produced

	In essence, `InfoFlow` allows us to treat raw data as a tensor, then operate on its dimensionality as split into parts whose transform varies _with_ the output (aka. a _covariant_ index), and parts whose transform varies _against_ the output (aka. _contravariant_ value).

	## Benefits
	The reasons to do this are numerous:

	- **Clarity**: Using `InfoFlow`, it's easy to understand what the data is, and what can be done to it, making it much easier to implement complex operations in math nodes without sacrificing the user's mental model.
	- **Zero-Cost Operations**: Transforming indices, "folding" dimensions into the output, and other such operations don't actually do anything to the data, enabling a lot of operations to feel "free" in terms of performance.
	- **Semantic Indexing**: Using `InfoFlow`, it's easy to index and slice arrays using ex. nanometer vacuum wavelengths, instead of arbitrary integers.
	"""

	####################
	# - Dimensions: Covariant Index
	####################
	dims: dict[sim_symbols.SimSymbol, IndexArray] = dataclasses.field(
		default_factory=dict
	)

	@functools.cached_property
	def last_dim(self) -> sim_symbols.SimSymbol | None:
		"""The integer axis occupied by the dimension.

		Can be used to index `.shape` of the represented raw array.
		"""
		if self.dims:
			return next(iter(self.dims.keys()))
		return None

	@functools.cached_property
	def last_dim(self) -> sim_symbols.SimSymbol | None:
		"""The integer axis occupied by the dimension.

		Can be used to index `.shape` of the represented raw array.
		"""
		if self.dims:
			return list(self.dims.keys())[-1]
		return None

	def dim_axis(self, dim: sim_symbols.SimSymbol) -> int:
		"""The integer axis occupied by the dimension.

		Can be used to index `.shape` of the represented raw array.
		"""
		return list(self.dims.keys()).index(dim)

	def has_idx_cont(self, dim: sim_symbols.SimSymbol) -> bool:
		"""Whether the dim's index is continuous, and therefore index array.

		This happens when the dimension is generated from a symbolic function, as opposed to from discrete observations.
		In these cases, the `SimSymbol.domain` of the dimension should be used to determine the overall domain of validity.

		Other than that, it's up to the user to select a particular way of indexing.
		"""
		return self.dims[dim] is None

	def has_idx_discrete(self, dim: sim_symbols.SimSymbol) -> bool:
		"""Whether the (rat|real) dim is indexed by an `ArrayFlow` / `RangeFlow`."""
		return isinstance(self.dims[dim], ArrayFlow | RangeFlow)

	def has_idx_labels(self, dim: sim_symbols.SimSymbol) -> bool:
		"""Whether the (int) dim is indexed by a `LabelArray`."""
		if dim.mathtype is spux.MathType.Integer:
			return isinstance(self.dims[dim], list)
		return False

	####################
	# - Output: Contravariant Value
	####################
	output: sim_symbols.SimSymbol

	####################
	# - Pinned Dimension Values
	####################
	## -> Whenever a dimension is deleted, we retain what that index value was.
	## -> This proves to be very helpful for clear visualization.
	pinned_values: dict[sim_symbols.SimSymbol, spux.SympyExpr] = dataclasses.field(
		default_factory=dict
	)

	####################
	# - Operations: Dimensions
	####################
	def prepend_dim(
		self, dim: sim_symbols.SimSymbol, dim_idx: sim_symbols.SimSymbol
	) -> typ.Self:
		"""Insert a new dimension at index 0."""
		return InfoFlow(
			dims={dim: dim_idx} | self.dims,
			output=self.output,
			pinned_values=self.pinned_values,
		)

	def slice_dim(
		self, dim: sim_symbols.SimSymbol, slice_tuple: tuple[int, int, int]
	) -> typ.Self:
		"""Slice a dimensional array by-index along a particular dimension."""
		return InfoFlow(
			dims={
				_dim: dim_idx[slice_tuple[0] : slice_tuple[1] : slice_tuple[2]]
				if _dim == dim
				else _dim
				for _dim, dim_idx in self.dims.items()
			},
			output=self.output,
			pinned_values=self.pinned_values,
		)

	def replace_dim(
		self,
		old_dim: sim_symbols.SimSymbol,
		new_dim: sim_symbols.SimSymbol,
		new_dim_idx: IndexArray,
	) -> typ.Self:
		"""Replace a dimension entirely, in-place, including symbol and index array."""
		return InfoFlow(
			dims={
				(new_dim if _dim == old_dim else _dim): (
					new_dim_idx if _dim == old_dim else _dim
				)
				for _dim, dim_idx in self.dims.items()
			},
			output=self.output,
			pinned_values=self.pinned_values,
		)

	def replace_dims(
		self, new_dims: dict[sim_symbols.SimSymbol, IndexArray]
	) -> typ.Self:
		"""Replace several dimensional indices with new index arrays/ranges."""
		return InfoFlow(
			dims={
				dim: new_dims.get(dim, dim_idx) for dim, dim_idx in self.dim_idx.items()
			},
			output=self.output,
			pinned_values=self.pinned_values,
		)

	def delete_dim(
		self, dim_to_remove: sim_symbols.SimSymbol, pin_idx: int | None = None
	) -> typ.Self:
		"""Delete a dimension, optionally pinning the value of an index from that dimension."""
		new_pin = (
			{dim_to_remove: self.dims[dim_to_remove][pin_idx]}
			if pin_idx is not None
			else {}
		)
		return InfoFlow(
			dims={
				dim: dim_idx
				for dim, dim_idx in self.dims.items()
				if dim != dim_to_remove
			},
			output=self.output,
			pinned_values=self.pinned_values | new_pin,
		)

	def swap_dimensions(self, dim_0: str, dim_1: str) -> typ.Self:
		"""Swap the positions of two dimensions."""

		# Swapped Dimension Keys
		def name_swapper(dim_name):
			return (
				dim_name
				if dim_name not in [dim_0, dim_1]
				else {dim_0: dim_1, dim_1: dim_0}[dim_name]
			)

		swapped_dim_keys = [name_swapper(dim) for dim in self.dims]

		return InfoFlow(
			dims={dim_key: self.dims[dim_key] for dim_key in swapped_dim_keys},
			output=self.output,
			pinned_values=self.pinned_values,
		)

	####################
	# - Operations: Output
	####################
	def update_output(self, **kwargs) -> typ.Self:
		"""Passthrough to `SimSymbol.update()` method on `self.output`."""
		return InfoFlow(
			dims=self.dims,
			output=self.output.update(**kwargs),
			pinned_values=self.pinned_values,
		)

	####################
	# - Operations: Fold
	####################
	def fold_last_input(self):
		"""Fold the last input dimension into the output."""
		last_key = list(self.dims.keys())[-1]
		last_idx = list(self.dims.values())[-1]

		rows = self.output.rows
		cols = self.output.cols
		match (rows, cols):
			case (1, 1):
				new_output = self.output.set_size(len(last_idx), 1)
			case (_, 1):
				new_output = self.output.set_size(rows, len(last_idx))
			case (1, _):
				new_output = self.output.set_size(len(last_idx), cols)
			case (_, _):
				raise NotImplementedError  ## Not yet :)

		return InfoFlow(
			dims={
				dim: dim_idx for dim, dim_idx in self.dims.items() if dim != last_key
			},
			output=new_output,
			pinned_values=self.pinned_values,
		)
