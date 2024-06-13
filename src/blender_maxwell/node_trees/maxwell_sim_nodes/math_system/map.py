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

"""Implements map operations for the `MapNode`."""

import enum
import typing as typ

import jax.numpy as jnp
import jaxtyping as jtyp
import sympy as sp
import sympy.physics.units as spu

from blender_maxwell.utils import logger, sim_symbols
from blender_maxwell.utils import sympy_extra as spux

from .. import contracts as ct

log = logger.get(__name__)

MT = spux.MathType
PT = spux.PhysicalType


class MapOperation(enum.StrEnum):
	"""Valid operations for the `MapMathNode`.

	Attributes:
		Real: Compute the real part of the input.
		Imag: Compute the imaginary part of the input.
		Abs: Compute the absolute value of the input.
		Sq: Square the input.
		Sqrt: Compute the (principal) square root of the input.
		InvSqrt: Compute the inverse square root of the input.
		Cos: Compute the cosine of the input.
		Sin: Compute the sine of the input.
		Tan: Compute the tangent of the input.
		Acos: Compute the inverse cosine of the input.
		Asin: Compute the inverse sine of the input.
		Atan: Compute the inverse tangent of the input.
		Norm2: Compute the 2-norm (aka. length) of the input vector.
		Det: Compute the determinant of the input matrix.
		Cond: Compute the condition number of the input matrix.
		NormFro: Compute the frobenius norm of the input matrix.
		Rank: Compute the rank of the input matrix.
		Diag: Compute the diagonal vector of the input matrix.
		EigVals: Compute the eigenvalues vector of the input matrix.
		SvdVals: Compute the singular values vector of the input matrix.
		Inv: Compute the inverse matrix of the input matrix.
		Tra: Compute the transpose matrix of the input matrix.
		QR_Q: Compute the QR-factorized matrices of the input matrix, and extract the 'Q' component.
		QR_R: Compute the QR-factorized matrices of the input matrix, and extract the 'R' component.
		Chol: Compute the Cholesky-factorized matrices of the input matrix.
		Svd: Compute the SVD-factorized matrices of the input matrix.
	"""

	# By Number
	Real = enum.auto()
	Imag = enum.auto()
	Abs = enum.auto()
	Sq = enum.auto()
	Reciprocal = enum.auto()
	Sqrt = enum.auto()
	InvSqrt = enum.auto()
	Cos = enum.auto()
	Sin = enum.auto()
	Tan = enum.auto()
	Acos = enum.auto()
	Asin = enum.auto()
	Atan = enum.auto()
	Sinc = enum.auto()
	# By Vector
	Norm2 = enum.auto()
	# By Matrix
	Det = enum.auto()
	Cond = enum.auto()
	NormFro = enum.auto()
	Rank = enum.auto()
	Diag = enum.auto()
	EigVals = enum.auto()
	SvdVals = enum.auto()
	Inv = enum.auto()
	Tra = enum.auto()
	QR_Q = enum.auto()
	QR_R = enum.auto()
	# Chol = enum.auto()
	# Svd = enum.auto()

	####################
	# - UI
	####################
	@staticmethod
	def to_name(value: typ.Self) -> str:
		"""A human-readable UI-oriented name."""
		MO = MapOperation
		return {
			# By Number
			MO.Real: 'ℝ(v)',
			MO.Imag: 'Im(v)',
			MO.Abs: '|v|',
			MO.Sq: 'v²',
			MO.Sqrt: '√v',
			MO.Reciprocal: '1/v',
			MO.InvSqrt: '1/√v',
			MO.Cos: 'cos v',
			MO.Sin: 'sin v',
			MO.Tan: 'tan v',
			MO.Acos: 'acos v',
			MO.Asin: 'asin v',
			MO.Atan: 'atan v',
			MO.Sinc: 'sinc v',
			# By Vector
			MO.Norm2: '||v||₂',
			# By Matrix
			MO.Det: 'det V',
			MO.Cond: 'κ(V)',
			MO.NormFro: '||V||_F',
			MO.Rank: 'rank V',
			MO.Diag: 'diag V',
			MO.EigVals: 'eigvals V',
			MO.SvdVals: 'svdvals V',
			MO.Inv: 'V⁻¹',
			MO.Tra: 'Vt',
			MO.QR_Q: 'qr[Q] V',
			MO.QR_R: 'qr[R] V',
			# MO.Chol: 'chol V',
			# MO.Svd: 'svd V',
		}[value]

	@staticmethod
	def to_icon(_: typ.Self) -> str:
		"""No icons."""
		return ''

	def bl_enum_element(self, i: int) -> ct.BLEnumElement:
		"""Given an integer index, generate an element that conforms to the requirements of `bpy.props.EnumProperty.items`."""
		MO = MapOperation
		return (
			str(self),
			MO.to_name(self),
			MO.to_name(self),
			MO.to_icon(self),
			i,
		)

	@staticmethod
	def bl_enum_elements(info: ct.InfoFlow) -> list[ct.BLEnumElement]:
		"""Generate a list of guaranteed-valid operations based on the passed `InfoFlow`.

		Returns a `bpy.props.EnumProperty.items`-compatible list.
		"""
		return [
			operation.bl_enum_element(i)
			for i, operation in enumerate(MapOperation.from_info(info))
		]

	####################
	# - Derivation
	####################
	@staticmethod
	def from_info(info: ct.InfoFlow) -> list[typ.Self]:
		"""Derive valid mapping operations from the `InfoFlow` of the operand."""
		MO = MapOperation
		ops = []

		# Real/Imag
		if info.output.mathtype is MT.Complex:
			ops += [MO.Real, MO.Imag]

		# Absolute Value
		ops += [MO.Abs]

		# Square
		ops += [MO.Sq]

		# Reciprocal
		if info.output.is_nonzero:
			ops += [MO.Reciprocal]

		# Square Root (Principal)
		ops += [MO.Sqrt]

		# Inverse Sqrt
		if info.output.is_nonzero:
			ops += [MO.InvSqrt]

		# Cos/Sin/Tan/Sinc
		if info.output.unit == spu.radian:
			ops += [MO.Cos, MO.Sin, MO.Tan, MO.Sinc]

		# Inverse Cos/Sin/Tan
		## -> Presume complex-extensions that aren't limited.
		if info.output.physical_type is PT.NonPhysical and info.output.unit is None:
			ops += [MO.Acos, MO.Asin, MO.Atan]

		# By Vector
		if info.output.shape_len == 1:
			ops += [MO.Norm2]

		# By Matrix
		if info.output.shape_len == 2:  # noqa: PLR2004
			if info.output.rows == info.output.cols:
				ops += [MO.Det]

			# Square Matrix
			if info.output.rows == info.output.cols:
				# Det
				ops += [MO.Det]

				# Diag
				ops += [MO.Diag]

				# Inv
				ops += [MO.Inv]

			# Cond
			ops += [MO.Cond]

			# NormFro
			ops += [MO.NormFro]

			# Rank
			ops += [MO.Rank]

			# EigVals
			ops += [MO.EigVals]

			# SvdVals
			ops += [MO.EigVals]

			# Tra
			ops += [MO.Tra]

			# QR
			ops += [MO.QR_Q, MO.QR_R]

		return ops

	####################
	# - Implementations
	####################
	@property
	def sp_func(self):
		"""Implement the mapping operation for sympy expressions."""
		MO = MapOperation
		return {
			# By Number
			MO.Real: lambda expr: sp.re(expr),
			MO.Imag: lambda expr: sp.im(expr),
			MO.Abs: lambda expr: sp.Abs(expr),
			MO.Sq: lambda expr: expr**2,
			MO.Sqrt: lambda expr: sp.sqrt(expr),
			MO.Reciprocal: lambda expr: 1 / expr,
			MO.InvSqrt: lambda expr: 1 / sp.sqrt(expr),
			MO.Cos: lambda expr: sp.cos(expr),
			MO.Sin: lambda expr: sp.sin(expr),
			MO.Tan: lambda expr: sp.tan(expr),
			MO.Acos: lambda expr: sp.acos(expr),
			MO.Asin: lambda expr: sp.asin(expr),
			MO.Atan: lambda expr: sp.atan(expr),
			MO.Sinc: lambda expr: sp.sinc(expr),
			# By Vector
			# Vector -> Number
			MO.Norm2: lambda expr: sp.sqrt(expr.T @ expr)[0],
			# By Matrix
			# Matrix -> Number
			MO.Det: lambda expr: sp.det(expr),
			MO.Cond: lambda expr: expr.condition_number(),
			MO.NormFro: lambda expr: expr.norm(ord='fro'),
			MO.Rank: lambda expr: expr.rank(),
			# Matrix -> Vec
			MO.Diag: lambda expr: expr.diagonal(),
			MO.EigVals: lambda expr: sp.ImmutableMatrix(list(expr.eigenvals().keys())),
			MO.SvdVals: lambda expr: expr.singular_values(),
			# Matrix -> Matrix
			MO.Inv: lambda expr: expr.inv(),
			MO.Tra: lambda expr: expr.T,
			# Matrix -> Matrices
			MO.QR_Q: lambda expr: expr.QRdecomposition()[0],
			MO.QR_R: lambda expr: expr.QRdecomposition()[1],
			# MO.Chol: lambda expr: expr.cholesky(),
			# MO.Svd: lambda expr: expr.singular_value_decomposition(),
		}[self]

	@property
	def jax_func(
		self,
	) -> typ.Callable[
		[jtyp.Shaped[jtyp.Array, '...'], int], jtyp.Shaped[jtyp.Array, '...']
	]:
		"""Implements the identified mapping using `jax`."""
		MO = MapOperation
		return {
			# By Number
			MO.Real: lambda expr: jnp.real(expr),
			MO.Imag: lambda expr: jnp.imag(expr),
			MO.Abs: lambda expr: jnp.abs(expr),
			MO.Sq: lambda expr: jnp.square(expr),
			MO.Sqrt: lambda expr: jnp.sqrt(expr),
			MO.Reciprocal: lambda expr: 1 / expr,
			MO.InvSqrt: lambda expr: 1 / jnp.sqrt(expr),
			MO.Cos: lambda expr: jnp.cos(expr),
			MO.Sin: lambda expr: jnp.sin(expr),
			MO.Tan: lambda expr: jnp.tan(expr),
			MO.Acos: lambda expr: jnp.acos(expr),
			MO.Asin: lambda expr: jnp.asin(expr),
			MO.Atan: lambda expr: jnp.atan(expr),
			MO.Sinc: lambda expr: jnp.sinc(expr),
			# By Vector
			# Vector -> Number
			MO.Norm2: lambda expr: jnp.linalg.norm(expr, ord=2, axis=-1),
			# By Matrix
			# Matrix -> Number
			MO.Det: lambda expr: jnp.linalg.det(expr),
			MO.Cond: lambda expr: jnp.linalg.cond(expr),
			MO.NormFro: lambda expr: jnp.linalg.matrix_norm(expr, ord='fro'),
			MO.Rank: lambda expr: jnp.linalg.matrix_rank(expr),
			# Matrix -> Vec
			MO.Diag: lambda expr: jnp.diagonal(expr, axis1=-2, axis2=-1),
			MO.EigVals: lambda expr: jnp.linalg.eigvals(expr),
			MO.SvdVals: lambda expr: jnp.linalg.svdvals(expr),
			# Matrix -> Matrix
			MO.Inv: lambda expr: jnp.linalg.inv(expr),
			MO.Tra: lambda expr: jnp.matrix_transpose(expr),
			# Matrix -> Matrices
			MO.QR_Q: lambda expr: jnp.linalg.qr(expr)[0],
			MO.QR_R: lambda expr: jnp.linalg.qr(expr, mode='r'),
			# MO.Chol: lambda expr: jnp.linalg.cholesky(expr),
			# MO.Svd: lambda expr: jnp.linalg.svd(expr),
		}[self]

	####################
	# - Transforms: FlowKind
	####################
	def transform_func(self, func: ct.FuncFlow) -> ct.FuncFlow:
		"""Transform input function according to the current operation and output info characterization."""
		return func.compose_within(
			self.jax_func,
			enclosing_func_output=self.transform_output(func.func_output),
			supports_jax=True,
		)

	def transform_info(self, info: ct.InfoFlow):
		"""Transform the `InfoFlow` characterizing the output."""
		return info.update(output=self.transform_output(info.output))

	def transform_params(self, params: ct.ParamsFlow):
		"""Transform the incoming function parameters to include output arguments."""
		return params

	####################
	# - Transforms: Symbolic
	####################
	def transform_output(self, sym: sim_symbols.SimSymbol):  # noqa: PLR0911
		"""Transform the `SimSymbol` characterizing the output."""
		MO = MapOperation

		dm = sym.domain
		match self:
			# By Number
			case MO.Real:
				return sym.update(
					mathtype=MT.Real,
					domain=dm.real,
				)
			case MO.Imag:
				return sym.update(
					mathtype=MT.Real,
					domain=dm.imag,
				)
			case MO.Abs:
				return sym.update(
					mathtype=MT.Real,
					domain=dm.abs,
				)

			case MO.Sq:
				return sym.update(
					domain=dm**2,
				)
			case MO.Reciprocal:
				orig_unit = sym.unit
				new_unit = 1 / orig_unit if orig_unit is not None else None
				new_phy_type = PT.from_unit(new_unit, optional=True)

				return sym.update(
					physical_type=new_phy_type,
					unit=new_unit,
					domain=dm.reciprocal,
				)
			case MO.Sqrt:
				## TODO: Complex -> Real MathType
				return sym.update(domain=dm ** sp.Rational(1, 2))
			case MO.InvSqrt:
				## TODO: Complex -> Real MathType
				return sym.update(domain=(dm ** sp.Rational(1, 2)).reciprocal)

			case MO.Cos:
				return sym.update(
					physical_type=PT.NonPhysical,
					unit=None,
					domain=dm.cos,
				)
			case MO.Sin:
				return sym.update(
					physical_type=PT.NonPhysical,
					unit=None,
					domain=dm.sin,
				)
			case MO.Tan:
				return sym.update(
					physical_type=PT.NonPhysical,
					unit=None,
					domain=dm.tan,
				)

			case MO.Acos:
				return sym.update(
					mathtype=MT.Complex if sym.mathtype is MT.Complex else MT.Real,
					physical_type=PT.Angle,
					unit=spu.radian,
					domain=dm.acos,
				)
			case MO.Asin:
				return sym.update(
					mathtype=MT.Complex if sym.mathtype is MT.Complex else MT.Real,
					physical_type=PT.Angle,
					unit=spu.radian,
					domain=dm.asin,
				)
			case MO.Atan:
				return sym.update(
					mathtype=MT.Complex if sym.mathtype is MT.Complex else MT.Real,
					physical_type=PT.Angle,
					unit=spu.radian,
					domain=dm.atan,
				)

			case MO.Sinc:
				return sym.update(
					physical_type=PT.NonPhysical,
					unit=None,
					domain=dm.sinc,
				)

			# By Vector/Covector
			case MO.Norm2:
				size = max([sym.rows, sym.cols])
				return sym.update(
					mathtype=MT.Real,
					rows=1,
					cols=1,
					domain=(size * dm**2) ** sp.Rational(1, 2),
				)

			# By Matrix
			case MO.Det:
				## -> NOTE: Determinant only valid for square matrices.
				size = sym.rows
				orig_unit = sym.unit

				new_unit = orig_unit**size if orig_unit is not None else None
				_new_phy_type = PT.from_unit(new_unit, optional=True)
				new_phy_type = (
					_new_phy_type if _new_phy_type is not None else PT.NonPhysical
				)

				return sym.update(
					physical_type=new_phy_type,
					unit=new_unit,
					rows=1,
					cols=1,
					domain=(size * dm**2) ** sp.Rational(1, 2),
				)

			case MO.Cond:
				return sym.update(
					mathtype=MT.Real,
					physical_type=PT.NonPhysical,
					unit=None,
					rows=1,
					cols=1,
					domain=spux.BlessedSet(sp.Interval(1, sp.oo)),
				)

			case MO.NormFro:
				return sym.update(
					mathtype=MT.Real,
					rows=1,
					cols=1,
					domain=(sym.rows * sym.cols * abs(dm) ** 2) ** sp.Rational(1, 2),
				)

			case MO.Rank:
				return sym.update(
					mathtype=MT.Integer,
					physical_type=PT.NonPhysical,
					unit=None,
					rows=1,
					cols=1,
					domain=spux.BlessedSet(sp.Range(0, min([sym.rows, sym.cols]) + 1)),
				)

			case MO.Diag:
				return sym.update(cols=1)

			case MO.EigVals:
				## TODO: Gershgorin circle theorem?
				return spux.BlessedSet(sp.Complexes)

			case MO.SvdVals:
				## TODO: Domain bound on singular values?
				## -- We might also consider a 'nonzero singvals' operation.
				## -- Since singular values can be zero just fine.
				return sym.update(
					mathtype=MT.Real,
					cols=1,
					domain=spux.BlessedSet(sp.Interval(0, sp.oo)),
				)

			case MO.Inv:
				## -> Defined: Square non-singular matrices.
				orig_unit = sym.unit
				new_unit = 1 / orig_unit if orig_unit is not None else None
				new_phy_type = PT.from_unit(new_unit, optional=True)

				return sym.update(
					physical_type=new_phy_type,
					unit=new_unit,
					domain=sym.mathtype.symbolic_set,
				)

			case MO.Tra:
				return sym.update(
					rows=sym.cols,
					cols=sym.rows,
				)

			case MO.QR_Q:
				return sym.update(
					mathtype=MT.Complex if sym.mathtype is MT.Complex else MT.Real,
					physical_type=PT.NonPhysical,
					unit=None,
					cols=min([sym.rows, sym.cols]),
					domain=(
						spux.BlessedSet(spux.ComplexRegion(sp.Interval(-1, 1) ** 2))
						if sym.mathtype is MT.Complex
						else spux.BlessedSet(sp.Interval(-1, 1))
					),
				)

			case MO.QR_R:
				return sym
