import dataclasses
import typing as typ
from types import MappingProxyType

import jax
import jax.numpy as jnp
import pandas as pd

# import jaxtyping as jtyp
import sympy.physics.units as spu
import xarray

from . import extra_sympy_units as spux
from . import logger

log = logger.get(__name__)

DimName: typ.TypeAlias = str
Number: typ.TypeAlias = int | float | complex
NumberRange: typ.TypeAlias = jax.Array


@dataclasses.dataclass(kw_only=True)
class JArray:
	"""Very simple wrapper for JAX arrays, which includes information about the dimension names and bounds."""

	array: jax.Array
	dims: dict[DimName, NumberRange]
	dim_units: dict[DimName, spu.Quantity]

	####################
	# - Constructor
	####################
	@classmethod
	def from_xarray(
		cls,
		xarr: xarray.DataArray,
		dim_units: dict[DimName, spu.Quantity] = MappingProxyType({}),
		sort_axis: int = -1,
	) -> typ.Self:
		return cls(
			array=jnp.sort(jnp.array(xarr.data), axis=sort_axis),
			dims={
				dim_name: jnp.array(xarr.get_index(dim_name).values)
				for dim_name in xarr.dims
			},
			dim_units={dim_name: dim_units.get(dim_name) for dim_name in xarr.dims},
		)

	def idx(self, dim_name: DimName, dim_value: Number) -> int:
		found_idx = jnp.searchsorted(self.dims[dim_name], dim_value)
		if found_idx == 0:
			return found_idx
		if found_idx == len(self.dims[dim_name]):
			return found_idx - 1

		left = self.dims[dim_name][found_idx - 1]
		right = self.dims[dim_name][found_idx - 1]
		return found_idx - 1 if (dim_value - left) <= (right - dim_value) else found_idx

	@property
	def dtype(self) -> jnp.dtype:
		return self.array.dtype
