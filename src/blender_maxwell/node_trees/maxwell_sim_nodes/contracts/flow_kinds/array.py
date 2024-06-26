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

import base64
import functools
import io
import typing as typ

import jax
import jax.numpy as jnp
import jaxtyping as jtyp
import numpy as np
import pydantic as pyd
import sympy as sp

from blender_maxwell.utils import logger
from blender_maxwell.utils import sympy_extra as spux
from blender_maxwell.utils.jaxarray import JaxArrayBytes
from blender_maxwell.utils.lru_method import method_lru

log = logger.get(__name__)


# TODO: Our handling of 'is_sorted' is sloppy and probably wrong.
class ArrayFlow(pyd.BaseModel):
	"""A homogeneous, realized array of numerical values with an optionally-attached unit and sort-tracking.

	While the principle is simple, arrays-with-units ends up being a powerful basis for derived and computed features/methods/processing.

	Attributes:
		values: An ND array-like object of arbitrary numerical type.
		unit: A `sympy` unit.
			None if unitless.
	"""

	model_config = pyd.ConfigDict(frozen=True)

	unit: spux.Unit | None = None
	is_sorted: bool = False

	####################
	# - Array Access
	####################
	jax_bytes: JaxArrayBytes  ## Immutable jax.Array, anyone? :)

	@functools.cached_property
	def values(self) -> jax.Array:
		"""Return the jax array."""
		with io.BytesIO() as memfile:
			memfile.write(base64.b64decode(self.jax_bytes.decode('utf-8')))
			memfile.seek(0)
			return jnp.load(memfile)

	####################
	# - Computed Properties
	####################
	@method_lru()
	def __len__(self) -> int:
		"""Outer length of the contained array."""
		return len(self.values)

	@functools.cached_property
	def mathtype(self) -> spux.MathType:
		"""Deduce the `spux.MathType` of the first element of the contained array.

		This is generally a heuristic, but because `jax` enforces homogeneous arrays, this is actually a well-defined approach.
		"""
		return spux.MathType.from_pytype(type(self.values.item(0)))

	@functools.cached_property
	def physical_type(self) -> spux.MathType:
		"""Deduce the `spux.PhysicalType` of the unit."""
		return spux.PhysicalType.from_unit(self.unit)

	####################
	# - Array Features
	####################
	@functools.cached_property
	def realize_array(self) -> jtyp.Shaped[jtyp.Array, '...']:
		"""Standardized access to `self.values`."""
		return self.values

	@functools.cached_property
	def shape(self) -> int:
		"""Shape of the contained array."""
		return self.values.shape

	@method_lru(maxsize=32)
	def _getitem_index(self, i: int) -> typ.Self | spux.SympyExpr:
		value = self.values[i]
		if len(value.shape) == 0:
			return value * self.unit if self.unit is not None else sp.S(value)
		return ArrayFlow(jax_bytes=value, unit=self.unit, is_sorted=self.is_sorted)

	def __getitem__(self, subscript: slice) -> typ.Self | spux.SympyExpr:
		"""Implement indexing and slicing in a sane way.

		- **Integer Index**: For scalar output, return a `sympy` expression of the scalar multiplied by the unit, else just a sympy expression of the value.
		- **Slice**: Slice the internal array directly, and wrap the result in a new `ArrayFlow`.
		"""
		if isinstance(subscript, slice):
			return ArrayFlow(
				jax_bytes=self.values[subscript],
				unit=self.unit,
				is_sorted=self.is_sorted,
			)

		if isinstance(subscript, int):
			return self._getitem_index(subscript)

		raise NotImplementedError

	####################
	# - Methods
	####################
	def rescale(
		self, rescale_func, reverse: bool = False, new_unit: spux.Unit | None = None
	) -> typ.Self:
		"""Apply an order-preserving function to each element of the array, then (optionally) transform the result w/new unit and/or order.

		An optimized expression will be built and applied to `self.values` using `sympy.lambdify()`.

		Parameters:
			rescale_func: An **order-preserving** function to apply to each array element.
			reverse: Whether to reverse the order of the result.
			new_unit: An (optional) new unit to scale the result to.
		"""
		# Compile JAX-Compatible Rescale Function
		## -> Generally, we try to keep things nice and rational.
		## -> However, too-large ints may cause JAX to suffer from an overflow.
		## -> Jax works in 32-bit domain by default, for performance.
		## -> While it can be adjusted, that would also have tradeoffs.
		## -> Instead, a quick float() turns all the big-ints into floats.
		## -> Not super satisfying, but hey - it's all numerical anyway.
		a = self.mathtype.sp_symbol_a
		rescale_expr = (
			spux.scale_to_unit(rescale_func(a * self.unit), new_unit)
			if self.unit is not None
			else rescale_func(a)
		)
		_rescale_func = sp.lambdify(a, rescale_expr.n(), 'jax')
		values = _rescale_func(self.values)

		# Return ArrayFlow
		return ArrayFlow(
			jax_bytes=values[::-1] if reverse else values,
			unit=new_unit,
			is_sorted=self.is_sorted,
		)

	# @method_lru()
	def nearest_idx_of(self, value: spux.SympyType, require_sorted: bool = True) -> int:
		"""Find the index of the value that is closest to the given value.

		Units are taken into account; the given value will be scaled to the internal unit before direct use.

		Parameters:
			require_sorted: Require that `self.values` be sorted, so that use of the faster binary-search algorithm is guaranteed.

		Returns:
			The index of `self.values` that is closest to the value `value`.
		"""
		if not require_sorted:
			raise NotImplementedError

		# Scale Given Value to Internal Unit
		scaled_value = spux.sympy_to_python(spux.scale_to_unit(value, self.unit))

		# BinSearch for "Right IDX"
		## >>> self.values[right_idx] > scaled_value
		## >>> self.values[right_idx - 1] < scaled_value
		right_idx = jnp.searchsorted(self.values, scaled_value, side='left')

		# Case: Right IDX is Boundary
		if right_idx == 0:
			return int(right_idx)
		if right_idx == len(self.values):
			return int(right_idx - 1)

		# Find Closest of [Right IDX - 1, Right IDX]
		left_val = self.values[right_idx - 1]
		right_val = self.values[right_idx]

		if (scaled_value - left_val) <= (right_val - scaled_value):
			return int(right_idx - 1)

		return int(right_idx)

	####################
	# - Unit Transforms
	####################
	@method_lru()
	def correct_unit(self, unit: spux.Unit) -> typ.Self:
		"""Simply replace the existing unit with the given one.

		Parameters:
			corrected_unit: The new unit to insert.
				**MUST** be associable with a well-defined `PhysicalType`.
		"""
		return ArrayFlow(jax_bytes=self.values, unit=unit, is_sorted=self.is_sorted)

	@method_lru()
	def rescale_to_unit(self, new_unit: spux.Unit | None) -> typ.Self:
		"""Rescale the `ArrayFlow` to be expressed in the given unit.

		Parameters:
			corrected_unit: The new unit to insert.
				**MUST** be associable with a well-defined `PhysicalType`.
		"""
		return self.rescale(lambda v: v, new_unit=new_unit)

	def rescale_to_unit_system(self, unit_system: spux.UnitSystem | None) -> typ.Self:
		if unit_system is None:
			return self.values

		return self.correct_unit(None).rescale(
			lambda v: spux.scale_to_unit_system(v * self.unit, unit_system),
			new_unit=spux.convert_to_unit_system(self.unit, unit_system),
		)
