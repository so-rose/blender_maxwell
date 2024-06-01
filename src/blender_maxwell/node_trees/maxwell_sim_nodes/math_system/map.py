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
import sympy as sp

from blender_maxwell.utils import logger, sim_symbols
from blender_maxwell.utils import sympy_extra as spux

from .. import contracts as ct

log = logger.get(__name__)


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
		Qr: Compute the QR-factorized matrices of the input matrix.
		Chol: Compute the Cholesky-factorized matrices of the input matrix.
		Svd: Compute the SVD-factorized matrices of the input matrix.
	"""

	# By Number
	Real = enum.auto()
	Imag = enum.auto()
	Abs = enum.auto()
	Sq = enum.auto()
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
	Qr = enum.auto()
	Chol = enum.auto()
	Svd = enum.auto()

	####################
	# - UI
	####################
	@staticmethod
	def to_name(value: typ.Self) -> str:
		"""A human-readable UI-oriented name for a physical type."""
		MO = MapOperation
		return {
			# By Number
			MO.Real: 'ℝ(v)',
			MO.Imag: 'Im(v)',
			MO.Abs: '|v|',
			MO.Sq: 'v²',
			MO.Sqrt: '√v',
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
			MO.Qr: 'qr V',
			MO.Chol: 'chol V',
			MO.Svd: 'svd V',
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

	####################
	# - Ops from Shape
	####################
	@staticmethod
	def by_expr_info(info: ct.InfoFlow) -> list[typ.Self]:
		## TODO: By info, not shape.
		## TODO: Check valid domains/mathtypes for some functions.
		MO = MapOperation
		element_ops = [
			MO.Real,
			MO.Imag,
			MO.Abs,
			MO.Sq,
			MO.Sqrt,
			MO.InvSqrt,
			MO.Cos,
			MO.Sin,
			MO.Tan,
			MO.Acos,
			MO.Asin,
			MO.Atan,
			MO.Sinc,
		]

		match (info.output.rows, info.output.cols):
			case (1, 1):
				return element_ops

			case (_, 1):
				return [*element_ops, MO.Norm2]

			case (rows, cols) if rows == cols:
				## TODO: Check hermitian/posdef for cholesky.
				## - Can we even do this with just the output symbol approach?
				return [
					*element_ops,
					MO.Det,
					MO.Cond,
					MO.NormFro,
					MO.Rank,
					MO.Diag,
					MO.EigVals,
					MO.SvdVals,
					MO.Inv,
					MO.Tra,
					MO.Qr,
					MO.Chol,
					MO.Svd,
				]

			case (rows, cols):
				return [
					*element_ops,
					MO.Cond,
					MO.NormFro,
					MO.Rank,
					MO.SvdVals,
					MO.Inv,
					MO.Tra,
					MO.Svd,
				]

		return []

	####################
	# - Function Properties
	####################
	@property
	def sp_func(self):
		MO = MapOperation
		return {
			# By Number
			MO.Real: lambda expr: sp.re(expr),
			MO.Imag: lambda expr: sp.im(expr),
			MO.Abs: lambda expr: sp.Abs(expr),
			MO.Sq: lambda expr: expr**2,
			MO.Sqrt: lambda expr: sp.sqrt(expr),
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
			MO.EigVals: lambda expr: sp.Matrix(list(expr.eigenvals().keys())),
			MO.SvdVals: lambda expr: expr.singular_values(),
			# Matrix -> Matrix
			MO.Inv: lambda expr: expr.inv(),
			MO.Tra: lambda expr: expr.T,
			# Matrix -> Matrices
			MO.Qr: lambda expr: expr.QRdecomposition(),
			MO.Chol: lambda expr: expr.cholesky(),
			MO.Svd: lambda expr: expr.singular_value_decomposition(),
		}[self]

	@property
	def jax_func(self):
		MO = MapOperation
		return {
			# By Number
			MO.Real: lambda expr: jnp.real(expr),
			MO.Imag: lambda expr: jnp.imag(expr),
			MO.Abs: lambda expr: jnp.abs(expr),
			MO.Sq: lambda expr: jnp.square(expr),
			MO.Sqrt: lambda expr: jnp.sqrt(expr),
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
			MO.Qr: lambda expr: jnp.linalg.qr(expr),
			MO.Chol: lambda expr: jnp.linalg.cholesky(expr),
			MO.Svd: lambda expr: jnp.linalg.svd(expr),
		}[self]

	def transform_info(self, info: ct.InfoFlow):
		MO = MapOperation

		return {
			# By Number
			MO.Real: lambda: info.update_output(mathtype=spux.MathType.Real),
			MO.Imag: lambda: info.update_output(mathtype=spux.MathType.Real),
			MO.Abs: lambda: info.update_output(mathtype=spux.MathType.Real),
			MO.Sq: lambda: info,
			MO.Sqrt: lambda: info,
			MO.InvSqrt: lambda: info,
			MO.Cos: lambda: info,
			MO.Sin: lambda: info,
			MO.Tan: lambda: info,
			MO.Acos: lambda: info,
			MO.Asin: lambda: info,
			MO.Atan: lambda: info,
			MO.Sinc: lambda: info,
			# By Vector
			MO.Norm2: lambda: info.update_output(
				mathtype=spux.MathType.Real,
				rows=1,
				cols=1,
				# Interval
				interval_finite_re=(0, sim_symbols.float_max),
				interval_inf=(False, True),
				interval_closed=(True, False),
			),
			# By Matrix
			MO.Det: lambda: info.update_output(
				rows=1,
				cols=1,
			),
			MO.Cond: lambda: info.update_output(
				mathtype=spux.MathType.Real,
				rows=1,
				cols=1,
				physical_type=spux.PhysicalType.NonPhysical,
				unit=None,
			),
			MO.NormFro: lambda: info.update_output(
				mathtype=spux.MathType.Real,
				rows=1,
				cols=1,
				# Interval
				interval_finite_re=(0, sim_symbols.float_max),
				interval_inf=(False, True),
				interval_closed=(True, False),
			),
			MO.Rank: lambda: info.update_output(
				mathtype=spux.MathType.Integer,
				rows=1,
				cols=1,
				physical_type=spux.PhysicalType.NonPhysical,
				unit=None,
				# Interval
				interval_finite_re=(0, sim_symbols.int_max),
				interval_inf=(False, True),
				interval_closed=(True, False),
			),
			# Matrix -> Vector  ## TODO: ALL OF THESE
			MO.Diag: lambda: info,
			MO.EigVals: lambda: info,
			MO.SvdVals: lambda: info,
			# Matrix -> Matrix  ## TODO: ALL OF THESE
			MO.Inv: lambda: info,
			MO.Tra: lambda: info,
			# Matrix -> Matrices  ## TODO: ALL OF THESE
			MO.Qr: lambda: info,
			MO.Chol: lambda: info,
			MO.Svd: lambda: info,
		}[self]()
