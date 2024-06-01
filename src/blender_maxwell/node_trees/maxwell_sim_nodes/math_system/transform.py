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

from blender_maxwell.utils import logger, sci_constants, sim_symbols
from blender_maxwell.utils import sympy_extra as spux

from .. import contracts as ct

log = logger.get(__name__)


class TransformOperation(enum.StrEnum):
	"""Valid operations for the `TransformMathNode`.

	Attributes:
		FreqToVacWL: Transform an frequency dimension to vacuum wavelength.
		VacWLToFreq: Transform a vacuum wavelength dimension to frequency.
		ConvertIdxUnit: Convert the unit of a dimension to a compatible unit.
		SetIdxUnit: Set all properties of a dimension.
		FirstColToFirstIdx: Extract the first data column and set the first dimension's index array equal to it.
			**For 2D integer-indexed data only**.

		IntDimToComplex: Fold a last length-2 integer dimension into the output, transforming it from a real-like type to complex type.
		DimToVec: Fold the last dimension into the scalar output, creating a vector output type.
		DimsToMat: Fold the last two dimensions into the scalar output, creating a matrix output type.
		FT: Compute the 1D fourier transform along a dimension.
			New dimensional bounds are computing using the Nyquist Limit.
			For higher dimensions, simply repeat along more dimensions.
		InvFT1D: Compute the inverse 1D fourier transform along a dimension.
			New dimensional bounds are computing using the Nyquist Limit.
			For higher dimensions, simply repeat along more dimensions.
	"""

	# Covariant Transform
	FreqToVacWL = enum.auto()
	VacWLToFreq = enum.auto()
	ConvertIdxUnit = enum.auto()
	SetIdxUnit = enum.auto()
	FirstColToFirstIdx = enum.auto()

	# Fold
	IntDimToComplex = enum.auto()
	DimToVec = enum.auto()
	DimsToMat = enum.auto()

	# Fourier
	FT1D = enum.auto()
	InvFT1D = enum.auto()

	# TODO: Affine
	## TODO

	####################
	# - UI
	####################
	@staticmethod
	def to_name(value: typ.Self) -> str:
		TO = TransformOperation
		return {
			# Covariant Transform
			TO.FreqToVacWL: '𝑓 → λᵥ',
			TO.VacWLToFreq: 'λᵥ → 𝑓',
			TO.ConvertIdxUnit: 'Convert Dim',
			TO.SetIdxUnit: 'Set Dim',
			TO.FirstColToFirstIdx: '1st Col → 1st Dim',
			# Fold
			TO.IntDimToComplex: '→ ℂ',
			TO.DimToVec: '→ Vector',
			TO.DimsToMat: '→ Matrix',
			## TODO: Vector to new last-dim integer
			## TODO: Matrix to two last-dim integers
			# Fourier
			TO.FT1D: 'FT',
			TO.InvFT1D: 'iFT',
		}[value]

	@property
	def name(self) -> str:
		return TransformOperation.to_name(self)

	@staticmethod
	def to_icon(_: typ.Self) -> str:
		return ''

	def bl_enum_element(self, i: int) -> ct.BLEnumElement:
		TO = TransformOperation
		return (
			str(self),
			TO.to_name(self),
			TO.to_name(self),
			TO.to_icon(self),
			i,
		)

	####################
	# - Methods
	####################
	def valid_dims(self, info: ct.InfoFlow) -> list[typ.Self]:
		TO = TransformOperation
		match self:
			case TO.FreqToVacWL:
				return [
					dim
					for dim in info.dims
					if dim.physical_type is spux.PhysicalType.Freq
				]

			case TO.VacWLToFreq:
				return [
					dim
					for dim in info.dims
					if dim.physical_type is spux.PhysicalType.Length
				]

			case TO.ConvertIdxUnit:
				return [
					dim
					for dim in info.dims
					if not info.has_idx_labels(dim)
					and spux.PhysicalType.from_unit(dim.unit, optional=True) is not None
				]

			case TO.SetIdxUnit:
				return [dim for dim in info.dims if not info.has_idx_labels(dim)]

			## ColDimToComplex: Implicit Last Dimension
			## DimToVec: Implicit Last Dimension
			## DimsToMat: Implicit Last 2 Dimensions

			case TO.FT1D | TO.InvFT1D:
				# Filter by Axis Uniformity
				## -> FT requires uniform axis (aka. must be RangeFlow).
				## -> NOTE: If FT isn't popping up, check ExtractDataNode.
				return [dim for dim in info.dims if info.is_idx_uniform(dim)]

		return []

	@staticmethod
	def by_info(info: ct.InfoFlow) -> list[typ.Self]:
		TO = TransformOperation
		operations = []

		# Covariant Transform
		## Freq -> VacWL
		if TO.FreqToVacWL.valid_dims(info):
			operations += [TO.FreqToVacWL]

		## VacWL -> Freq
		if TO.VacWLToFreq.valid_dims(info):
			operations += [TO.VacWLToFreq]

		## Convert Index Unit
		if TO.ConvertIdxUnit.valid_dims(info):
			operations += [TO.ConvertIdxUnit]

		if TO.SetIdxUnit.valid_dims(info):
			operations += [TO.SetIdxUnit]

		## Column to First Index (Array)
		if (
			len(info.dims) == 2  # noqa: PLR2004
			and info.first_dim.mathtype is spux.MathType.Integer
			and info.last_dim.mathtype is spux.MathType.Integer
			and info.output.shape_len == 0
		):
			operations += [TO.FirstColToFirstIdx]

		# Fold
		## Last Dim -> Complex
		if (
			len(info.dims) >= 1
			and (
				info.output.mathtype
				in [spux.MathType.Integer, spux.MathType.Rational, spux.MathType.Real]
			)
			and info.last_dim.mathtype is spux.MathType.Integer
			and info.has_idx_labels(info.last_dim)
			and len(info.dims[info.last_dim]) == 2  # noqa: PLR2004
		):
			operations += [TO.IntDimToComplex]

		## Last Dim -> Vector
		if len(info.dims) >= 1 and info.output.shape_len == 0:
			operations += [TO.DimToVec]

		## Last Dim -> Matrix
		if len(info.dims) >= 2 and info.output.shape_len == 0:  # noqa: PLR2004
			operations += [TO.DimsToMat]

		# Fourier
		if TO.FT1D.valid_dims(info):
			operations += [TO.FT1D]

		if TO.InvFT1D.valid_dims(info):
			operations += [TO.InvFT1D]

		return operations

	####################
	# - Function Properties
	####################
	def jax_func(self, axis: int | None = None):
		TO = TransformOperation
		return {
			# Covariant Transform
			## -> Freq <-> WL is a rescale (noop) AND flip (not noop).
			TO.FreqToVacWL: lambda expr: jnp.flip(expr, axis=axis),
			TO.VacWLToFreq: lambda expr: jnp.flip(expr, axis=axis),
			TO.ConvertIdxUnit: lambda expr: expr,
			TO.SetIdxUnit: lambda expr: expr,
			TO.FirstColToFirstIdx: lambda expr: jnp.delete(expr, 0, axis=1),
			# Fold
			## -> To Complex: This should generally be a no-op.
			TO.IntDimToComplex: lambda expr: jnp.squeeze(
				expr.view(dtype=jnp.complex64), axis=-1
			),
			TO.DimToVec: lambda expr: expr,
			TO.DimsToMat: lambda expr: expr,
			# Fourier
			TO.FT1D: lambda expr: jnp.fft(expr, axis=axis),
			TO.InvFT1D: lambda expr: jnp.ifft(expr, axis=axis),
		}[self]

	def transform_info(
		self,
		info: ct.InfoFlow,
		dim: sim_symbols.SimSymbol | None = None,
		data_col: jtyp.Shaped[jtyp.Array, ' size'] | None = None,
		new_dim_name: str | None = None,
		unit: spux.Unit | None = None,
		physical_type: spux.PhysicalType | None = None,
	) -> ct.InfoFlow:
		TO = TransformOperation
		return {
			# Covariant Transform
			TO.FreqToVacWL: lambda: info.replace_dim(
				(f_dim := dim),
				sim_symbols.wl(unit),
				info.dims[f_dim].rescale(
					lambda el: sci_constants.vac_speed_of_light / el,
					reverse=True,
					new_unit=unit,
				),
			),
			TO.VacWLToFreq: lambda: info.replace_dim(
				(wl_dim := dim),
				sim_symbols.freq(unit),
				info.dims[wl_dim].rescale(
					lambda el: sci_constants.vac_speed_of_light / el,
					reverse=True,
					new_unit=unit,
				),
			),
			TO.ConvertIdxUnit: lambda: info.replace_dim(
				dim,
				dim.update(unit=unit),
				(
					info.dims[dim].rescale_to_unit(unit)
					if info.has_idx_discrete(dim)
					else None  ## Continuous -- dim SimSymbol already scaled
				),
			),
			TO.SetIdxUnit: lambda: info.replace_dim(
				dim,
				dim.update(
					sym_name=new_dim_name,
					physical_type=physical_type,
					unit=unit,
				),
				(
					info.dims[dim].correct_unit(unit)
					if info.has_idx_discrete(dim)
					else None  ## Continuous -- dim SimSymbol already scaled
				),
			),
			TO.FirstColToFirstIdx: lambda: info.replace_dim(
				info.first_dim,
				info.first_dim.update(
					sym_name=new_dim_name,
					mathtype=spux.MathType.from_jax_array(data_col),
					physical_type=physical_type,
					unit=unit,
				),
				ct.RangeFlow.try_from_array(ct.ArrayFlow(values=data_col, unit=unit)),
			).slice_dim(info.last_dim, (1, len(info.dims[info.last_dim]), 1)),
			# Fold
			TO.IntDimToComplex: lambda: info.delete_dim(info.last_dim).update_output(
				mathtype=spux.MathType.Complex
			),
			TO.DimToVec: lambda: info.fold_last_input(),
			TO.DimsToMat: lambda: info.fold_last_input().fold_last_input(),
			# Fourier
			TO.FT1D: lambda: info.replace_dim(
				dim,
				[
					# FT'ed Unit: Reciprocal of the Original Unit
					dim.update(
						unit=1 / dim.unit if dim.unit is not None else 1
					),  ## TODO: Okay to not scale interval?
					# FT'ed Bounds: Reciprocal of the Original Unit
					info.dims[dim].bound_fourier_transform,
				],
			),
			TO.InvFT1D: lambda: info.replace_dim(
				info.last_dim,
				[
					# FT'ed Unit: Reciprocal of the Original Unit
					dim.update(
						unit=1 / dim.unit if dim.unit is not None else 1
					),  ## TODO: Okay to not scale interval?
					# FT'ed Bounds: Reciprocal of the Original Unit
					## -> Note the midpoint may revert to 0.
					## -> See docs for `RangeFlow.bound_inv_fourier_transform` for more.
					info.dims[dim].bound_inv_fourier_transform,
				],
			),
		}[self]()
