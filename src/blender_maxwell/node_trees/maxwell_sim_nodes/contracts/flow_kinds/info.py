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

from blender_maxwell.utils import logger, sim_symbols
from blender_maxwell.utils import sympy_extra as spux

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

	# Access
	@functools.cached_property
	def first_dim(self) -> sim_symbols.SimSymbol | None:
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

	def dim_by_idx(self, idx: int) -> sim_symbols.SimSymbol | None:
		"""Retrieve the dimension associated with a particular index."""
		if idx > 0 and idx < len(self.dims) - 1:
			return list(self.dims.keys())[idx]
		return None

	def dim_by_name(self, dim_name: str, optional: bool = False) -> int | None:
		"""The integer axis occupied by the dimension.

		Can be used to index `.shape` of the represented raw array.
		"""
		dims_with_name = [dim for dim in self.dims if dim.name == dim_name]
		if len(dims_with_name) == 1:
			return dims_with_name[0]

		if optional:
			return None

		msg = f'Dim name {dim_name} not found in InfoFlow (or >1 found)'
		raise ValueError(msg)

	# Information By-Dim
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

	def is_idx_uniform(self, dim: sim_symbols.SimSymbol) -> bool:
		"""Whether the given dim has explicitly uniform indexing.

		This is needed primarily to check whether a Fourier Transform can be meaningfully performed on the data over the dimension's axis.

		In practice, we've decided that only `RangeFlow` really truly _guarantees_ uniform indexing.
		While `ArrayFlow` may be uniform in practice, it's a very expensive to check, and it's far better to enforce that the user perform that check and opt for a `RangeFlow` instead, at the time of dimension definition.
		"""
		dim_idx = self.dims[dim]
		return isinstance(dim_idx, RangeFlow) and dim_idx.scaling == 'lin'

	def dim_axis(self, dim: sim_symbols.SimSymbol) -> int:
		"""The integer axis occupied by the dimension.

		Can be used to index `.shape` of the represented raw array.
		"""
		return list(self.dims.keys()).index(dim)

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
	# - Properties
	####################
	@functools.cached_property
	def input_mathtypes(self) -> tuple[spux.MathType, ...]:
		return tuple([dim.mathtype for dim in self.dims])

	@functools.cached_property
	def output_mathtypes(self) -> tuple[spux.MathType, int, int]:
		return [self.output.mathtype for _ in range(len(self.output.shape) + 1)]

	@functools.cached_property
	def order(self) -> tuple[spux.MathType, ...]:
		r"""The order of the tensor represented by this info.

		While that sounds fancy and all, it boils down to:

		$$
			|\texttt{dims}| + |\texttt{output}.\texttt{shape}|
		$$

		Doing so characterizes the full dimensionality of the tensor, which also perfectly matches the length of the raw data's shape.

		Notes:
			Corresponds to `len(raw_data.shape)`, if `raw_data` is the n-dimensional array corresponding to this `InfoFlow`.
		"""
		return len(self.dims) + self.output_shape_len

	@functools.cached_property
	def is_scalar(self) -> tuple[spux.MathType, int, int]:
		"""Whether the described object can be described as "scalar".

		True when `self.order == 0`.
		"""
		return self.order == 0

	####################
	# - Properties
	####################
	@functools.cached_property
	def dim_labels(self) -> dict[str, dict[str, str]]:
		"""Return a dictionary mapping pretty dim names to information oriented for columnar information display."""
		return {
			dim.name_pretty: {
				'length': str(len(dim_idx)) if dim_idx is not None else '∞',
				'mathtype': dim.mathtype_size_label,
				'unit': dim.unit_label,
			}
			for dim, dim_idx in self.dims.items()
		}

	####################
	# - Operations: Comparison
	####################
	def compare_dims_identical(self, other: typ.Self) -> bool:
		"""Whether that the quantity and properites of all dimension `SimSymbol`s are "identical".

		"Identical" is defined according to the semantics of `SimSymbol.compare()`, which generally means that everything but the exact name and unit are different.
		"""
		return len(self.dims) == len(other.dims) and all(
			dim_l.compare(dim_r)
			for dim_l, dim_r in zip(self.dims, other.dims, strict=True)
		)

	def compare_addable(
		self, other: typ.Self, allow_differing_unit: bool = False
	) -> bool:
		"""Whether the two `InfoFlows` can be added/subtracted elementwise.

		Parameters:
			allow_differing_unit: When set,
				Forces the user to be explicit about specifying
		"""
		return self.compare_dims_identical(other) and self.output.compare_addable(
			other.output, allow_differing_unit=allow_differing_unit
		)

	def compare_multiplicable(self, other: typ.Self) -> bool:
		"""Whether the two `InfoFlow`s can be multiplied (elementwise).

		- The output `SimSymbol`s must be able to be multiplied.
		- Either the LHS is a scalar, the RHS is a scalar, or the dimensions are identical.
		"""
		return self.output.compare_multiplicable(other.output) and (
			(len(self.dims) == 0 and self.output.shape_len == 0)
			or (len(other.dims) == 0 and other.output.shape_len == 0)
			or self.compare_dims_identical(other)
		)

	def compare_exponentiable(self, other: typ.Self) -> bool:
		"""Whether the two `InfoFlow`s can be exponentiated.

		In general, we follow the rules of the "Hadamard Power" operator, which is also in use in `numpy` broadcasting rules.

		- The output `SimSymbol`s must be able to be exponentiated (mainly, the exponent can't have a unit).
		- Either the LHS is a scalar, the RHS is a scalar, or the dimensions are identical.
		"""
		return self.output.compare_exponentiable(other.output) and (
			(len(self.dims) == 0 and self.output.shape_len == 0)
			or (len(other.dims) == 0 and other.output.shape_len == 0)
			or self.compare_dims_identical(other)
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
				_dim: (
					dim_idx[slice_tuple[0] : slice_tuple[1] : slice_tuple[2]]
					if _dim == dim
					else dim_idx
				)
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
					new_dim_idx if _dim == old_dim else dim_idx
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

	def operate_output(
		self,
		other: typ.Self,
		op: typ.Callable[[spux.SympyExpr, spux.SympyExpr], spux.SympyExpr],
		unit_op: typ.Callable[[spux.SympyExpr, spux.SympyExpr], spux.SympyExpr],
	) -> spux.SympyExpr:
		"""Apply an operation between two the values and units of two `InfoFlow`s by reconstructing the properties of the new output `SimSymbol`."""
		sym_name = sim_symbols.SimSymbolName.Expr
		expr = op(self.output.sp_symbol_phy, other.output.sp_symbol_phy)
		unit_expr = unit_op(self.output.unit_factor, other.output.unit_factor)
		## TODO: Handle per-cell matrix units?

		return InfoFlow(
			dims=self.dims,
			output=sim_symbols.SimSymbol.from_expr(sym_name, expr, unit_expr),
			pinned_values=self.pinned_values,
		)

	####################
	# - Operations: Fold
	####################
	def fold_last_input(self):
		"""Fold the last input dimension into the output."""
		last_idx = self.dims[self.last_dim]

		rows = self.output.rows
		cols = self.output.cols
		match (rows, cols):
			case (1, 1):
				new_output = self.output.update(rows=len(last_idx), cols=1)
			case (_, 1):
				new_output = self.output.update(rows=rows, cols=len(last_idx))
			case (1, _):
				new_output = self.output.update(rows=len(last_idx), cols=cols)
			case (_, _):
				raise NotImplementedError  ## Not yet :)

		return InfoFlow(
			dims={
				dim: dim_idx
				for dim, dim_idx in self.dims.items()
				if dim != self.last_dim
			},
			output=new_output,
			pinned_values=self.pinned_values,
		)
