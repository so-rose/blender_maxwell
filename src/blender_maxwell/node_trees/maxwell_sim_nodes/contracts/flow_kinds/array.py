import dataclasses
import functools
import typing as typ

import jaxtyping as jtyp
import numpy as np
import sympy.physics.units as spu

from blender_maxwell.utils import extra_sympy_units as spux
from blender_maxwell.utils import logger

log = logger.get(__name__)


@dataclasses.dataclass(frozen=True, kw_only=True)
class ArrayFlow:
	"""A simple, flat array of values with an optionally-attached unit.

	Attributes:
		values: An ND array-like object of arbitrary numerical type.
		unit: A `sympy` unit.
			None if unitless.
	"""

	values: jtyp.Shaped[jtyp.Array, '...']
	unit: spux.Unit | None = None

	is_sorted: bool = False

	def __len__(self) -> int:
		return len(self.values)

	@functools.cached_property
	def mathtype(self) -> spux.MathType:
		return spux.MathType.from_pytype(type(self.values.item(0)))

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
		right_idx = np.searchsorted(self.values, scaled_value, side='left')

		# Case: Right IDX is Boundary
		if right_idx == 0:
			return right_idx
		if right_idx == len(self.values):
			return right_idx - 1

		# Find Closest of [Right IDX - 1, Right IDX]
		left_val = self.values[right_idx - 1]
		right_val = self.values[right_idx]

		if (scaled_value - left_val) <= (right_val - scaled_value):
			return right_idx - 1

		return right_idx

	def correct_unit(self, corrected_unit: spu.Quantity) -> typ.Self:
		if self.unit is not None:
			return ArrayFlow(
				values=self.values, unit=corrected_unit, is_sorted=self.is_sorted
			)

		msg = f'Tried to correct unit of unitless LazyDataValueRange "{corrected_unit}"'
		raise ValueError(msg)

	def rescale_to_unit(self, unit: spu.Quantity) -> typ.Self:
		if self.unit is not None:
			return ArrayFlow(
				values=float(spux.scaling_factor(self.unit, unit)) * self.values,
				unit=unit,
				is_sorted=self.is_sorted,  ## TODO: Can we really say that?
			)
			## TODO: Is this scaling numerically stable?

		msg = f'Tried to rescale unitless LazyDataValueRange to unit {unit}'
		raise ValueError(msg)

	def rescale_to_unit_system(self, unit: spu.Quantity) -> typ.Self:
		raise NotImplementedError
