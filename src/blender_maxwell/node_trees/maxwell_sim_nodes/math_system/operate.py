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
import sympy.physics.quantum as spq
import sympy.physics.units as spu

from blender_maxwell.utils import logger, sim_symbols
from blender_maxwell.utils import sympy_extra as spux

from .. import contracts as ct

log = logger.get(__name__)


def hadamard_power(lhs: spux.SympyType, rhs: spux.SympyType) -> spux.SympyType:
	"""Implement the Hadamard Power.

	Follows the specification in <https://docs.sympy.org/latest/modules/matrices/expressions.html#sympy.matrices.expressions.HadamardProduct>, which also conforms to `numpy` broadcasting rules for `**` on `np.ndarray`.
	"""
	match (isinstance(lhs, sp.MatrixBase), isinstance(rhs, sp.MatrixBase)):
		case (False, False):
			msg = f"Hadamard Power for two scalars is valid, but shouldn't be used - use normal power instead: {lhs} | {rhs}"
			raise ValueError(msg)

		case (True, False):
			return lhs.applyfunc(lambda el: el**rhs)

		case (False, True):
			return rhs.applyfunc(lambda el: lhs**el)

		case (True, True) if lhs.shape == rhs.shape:
			common_shape = lhs.shape
			return sp.ImmutableMatrix(
				*common_shape, lambda i, j: lhs[i, j] ** rhs[i, j]
			)

		case _:
			msg = f'Incompatible lhs and rhs for hadamard power: {lhs} | {rhs}'
			raise ValueError(msg)


class BinaryOperation(enum.StrEnum):
	"""Valid operations for the `OperateMathNode`.

	Attributes:
		Mul: Scalar multiplication.
		Div: Scalar division.
		Pow: Scalar exponentiation.
		Add: Elementwise addition.
		Sub: Elementwise subtraction.
		HadamMul: Elementwise multiplication (hadamard product).
		HadamPow: Principled shape-aware exponentiation (hadamard power).
		Atan2: Quadrant-respecting 2D arctangent.
		VecVecDot: Dot product for identically shaped vectors w/transpose.
		Cross: Cross product between identically shaped 3D vectors.
		VecVecOuter: Vector-vector outer product.
		LinSolve: Solve a linear system.
		LsqSolve: Minimize error of an underdetermined linear system.
		VecMatOuter: Vector-matrix outer product.
		MatMatDot: Matrix-matrix dot product.
	"""

	# Number | Number
	Mul = enum.auto()
	Div = enum.auto()
	Pow = enum.auto()

	# Elements | Elements
	Add = enum.auto()
	Sub = enum.auto()
	HadamMul = enum.auto()
	HadamPow = enum.auto()
	HadamDiv = enum.auto()
	Atan2 = enum.auto()

	# Vector | Vector
	VecVecDot = enum.auto()
	Cross = enum.auto()
	VecVecOuter = enum.auto()

	# Matrix | Vector
	LinSolve = enum.auto()
	LsqSolve = enum.auto()

	# Vector | Matrix
	VecMatOuter = enum.auto()

	# Matrix | Matrix
	MatMatDot = enum.auto()

	####################
	# - UI
	####################
	@staticmethod
	def to_name(value: typ.Self) -> str:
		"""A human-readable UI-oriented name for a physical type."""
		BO = BinaryOperation
		return {
			# Number | Number
			BO.Mul: 'ℓ · r',
			BO.Div: 'ℓ / r',
			BO.Pow: 'ℓ ^ r',  ## Also for square-matrix powers.
			# Elements | Elements
			BO.Add: 'ℓ + r',
			BO.Sub: 'ℓ - r',
			BO.HadamMul: '𝐋 ⊙ 𝐑',
			BO.HadamDiv: '𝐋 ⊙/ 𝐑',
			BO.HadamPow: '𝐥 ⊙^ 𝐫',
			BO.Atan2: 'atan2(ℓ:x, r:y)',
			# Vector | Vector
			BO.VecVecDot: '𝐥 · 𝐫',
			BO.Cross: 'cross(𝐥,𝐫)',
			BO.VecVecOuter: '𝐥 ⊗ 𝐫',
			# Matrix | Vector
			BO.LinSolve: '𝐋 ∖ 𝐫',
			BO.LsqSolve: 'argminₓ∥𝐋𝐱−𝐫∥₂',
			# Vector | Matrix
			BO.VecMatOuter: '𝐋 ⊗ 𝐫',
			# Matrix | Matrix
			BO.MatMatDot: '𝐋 · 𝐑',
		}[value]

	@staticmethod
	def to_icon(value: typ.Self) -> str:
		"""No icons."""
		return ''

	def bl_enum_element(self, i: int) -> ct.BLEnumElement:
		"""Given an integer index, generate an element that conforms to the requirements of `bpy.props.EnumProperty.items`."""
		BO = BinaryOperation
		return (
			str(self),
			BO.to_name(self),
			BO.to_name(self),
			BO.to_icon(self),
			i,
		)

	def bl_enum_elements(
		self, info_l: ct.InfoFlow, info_r: ct.InfoFlow
	) -> list[ct.BLEnumElement]:
		"""Generate a list of guaranteed-valid operations based on the passed `InfoFlow`s.

		Returns a `bpy.props.EnumProperty.items`-compatible list.
		"""
		return [
			operation.bl_enum_element(i)
			for i, operation in enumerate(BinaryOperation.by_infos(info_l, info_r))
		]

	####################
	# - Ops from Shape
	####################
	@staticmethod
	def by_infos(info_l: ct.InfoFlow, info_r: ct.InfoFlow) -> list[typ.Self]:
		"""Deduce valid binary operations from the shapes of the inputs."""
		BO = BinaryOperation
		ops = []

		# Add/Sub
		if info_l.compare_addable(info_r, allow_differing_unit=True):
			ops += [BO.Add, BO.Sub]

		# Mul/Div
		## -> Mul is ambiguous; we differentiate Hadamard and Standard.
		## -> Div additionally requires non-zero guarantees.
		if info_l.compare_multiplicable(info_r):
			match (info_l.order, info_r.order, info_r.output.is_nonzero):
				case (ordl, ordr, True) if ordl == 0 and ordr == 0:
					ops += [BO.Mul, BO.Div]
				case (ordl, ordr, True) if ordl > 0 and ordr == 0:
					ops += [BO.Mul, BO.Div]
				case (ordl, ordr, True) if ordl == 0 and ordr > 0:
					ops += [BO.Mul]
				case (ordl, ordr, True) if ordl > 0 and ordr > 0:
					ops += [BO.HadamMul, BO.HadamDiv]

				case (ordl, ordr, False) if ordl == 0 and ordr == 0:
					ops += [BO.Mul]
				case (ordl, ordr, False) if ordl > 0 and ordr == 0:
					ops += [BO.Mul]
				case (ordl, ordr, True) if ordl == 0 and ordr > 0:
					ops += [BO.Mul]
				case (ordl, ordr, False) if ordl > 0 and ordr > 0:
					ops += [BO.HadamMul]

		# Pow
		## -> We distinguish between "Hadamard Power" and "Power".
		## -> For scalars, they are the same (but we only expose "power").
		## -> For matrices, square matrices can be exp'ed by int powers.
		## -> Any other combination is well-defined by the Hadamard Power.
		if info_l.compare_exponentiable(info_r):
			match (info_l.order, info_r.order, info_r.output.mathtype):
				case (ordl, ordr, _) if ordl == 0 and ordr == 0:
					ops += [BO.Pow]

				case (ordl, ordr, spux.MathType.Integer) if (
					ordl > 0 and ordr == 0 and info_l.output.rows == info_l.output.cols
				):
					ops += [BO.Pow, BO.HadamPow]

				case _:
					ops += [BO.HadamPow]

		# Operations by-Output Length
		match (
			info_l.output.shape_len,
			info_r.output.shape_len,
		):
			# Number | Number
			case (0, 0) if info_l.is_scalar and info_r.is_scalar:
				# atan2: PhysicalType Must Both be Length | NonPhysical
				## -> atan2() produces radians from Cartesian coordinates.
				## -> This wouldn't make sense on non-Length / non-Unitless.
				if (
					info_l.output.physical_type is spux.PhysicalType.Length
					and info_r.output.physical_type is spux.PhysicalType.Length
				) or (
					info_l.output.physical_type is spux.PhysicalType.NonPhysical
					and info_l.output.unit is None
					and info_r.output.physical_type is spux.PhysicalType.NonPhysical
					and info_r.output.unit is None
				):
					ops += [BO.Atan2]

				return ops

			# Vector | Vector
			case (1, 1) if info_l.compare_dims_identical(info_r):
				outl = info_l.output
				outr = info_r.output

				# 1D Orders: Outer Product is Valid
				## -> We can't do per-element outer product.
				## -> However, it's still super useful on its own.
				if info_l.order == 1 and info_r.order == 1:
					ops += [BO.VecVecOuter]

				# Vector | Vector
				if outl.rows > outl.cols and outr.rows > outr.cols:
					ops += [BO.VecVecDot]

				# Covector | Vector
				if outl.rows < outl.cols and outr.rows > outr.cols:
					ops += [BO.MatMatDot]

				# Vector | Covector
				if outl.rows > outl.cols and outr.rows < outr.cols:
					ops += [BO.MatMatDot]

				# Covector | Covector
				if outl.rows < outl.cols and outr.rows < outr.cols:
					ops += [BO.VecVecDot]

				# Cross Product
				## -> Works great element-wise.
				## -> Enforce that both are 3x1 or 1x3.
				## -> See https://docs.sympy.org/latest/modules/matrices/matrices.html#sympy.matrices.matrices.MatrixBase.cross
				if (outl.rows == 3 and outr.rows == 3) or (
					outl.cols == 3 and outl.cols == 3
				):
					ops += [BO.Cross]

			# Vector | Matrix
			## -> We can't do per-element outer product.
			## -> However, it's still super useful on its own.
			case (1, 2) if info_l.compare_dims_identical(
				info_r
			) and info_l.order == 1 and info_r.order == 2:
				ops += [BO.VecMatOuter]

			# Matrix | Vector
			case (2, 1) if info_l.compare_dims_identical(info_r):
				# Mat-Vec Dot: Enforce RHS Column Vector
				if outr.rows > outl.cols:
					ops += [BO.MatMatDot]

				ops += [BO.LinSolve, BO.LsqSolve]

			## Matrix | Matrix
			case (2, 2):
				ops += [BO.MatMatDot]

		return ops

	####################
	# - Function Properties
	####################
	@property
	def sp_func(self):
		"""Deduce an appropriate sympy-based function that implements the binary operation for symbolic inputs."""
		BO = BinaryOperation

		## TODO: Make this compatible with sp.Matrix inputs
		return {
			# Number | Number
			BO.Mul: lambda exprs: exprs[0] * exprs[1],
			BO.Div: lambda exprs: exprs[0] / exprs[1],
			BO.Pow: lambda exprs: exprs[0] ** exprs[1],
			# Elements | Elements
			BO.Add: lambda exprs: exprs[0] + exprs[1],
			BO.Sub: lambda exprs: exprs[0] - exprs[1],
			BO.HadamMul: lambda exprs: exprs[0].multiply_elementwise(exprs[1]),
			BO.HadamPow: lambda exprs: sp.HadamardPower(exprs[0], exprs[1]),
			BO.Atan2: lambda exprs: sp.atan2(exprs[1], exprs[0]),
			# Vector | Vector
			BO.VecVecDot: lambda exprs: (exprs[0].T @ exprs[1])[0],
			BO.Cross: lambda exprs: exprs[0].cross(exprs[1]),
			BO.VecVecOuter: lambda exprs: exprs[0] @ exprs[1].T,
			# Matrix | Vector
			BO.LinSolve: lambda exprs: exprs[0].solve(exprs[1]),
			BO.LsqSolve: lambda exprs: exprs[0].solve_least_squares(exprs[1]),
			# Vector | Matrix
			BO.VecMatOuter: lambda exprs: spq.TensorProduct(exprs[0], exprs[1]),
			# Matrix | Matrix
			BO.MatMatDot: lambda exprs: exprs[0] @ exprs[1],
		}[self]

	@property
	def unit_func(self):
		"""The binary function to apply to both unit expressions, in order to deduce the unit expression of the output."""
		BO = BinaryOperation

		## TODO: Make this compatible with sp.Matrix inputs
		return {
			# Number | Number
			BO.Mul: BO.Mul.sp_func,
			BO.Div: BO.Div.sp_func,
			BO.Pow: BO.Pow.sp_func,
			# Elements | Elements
			BO.Add: BO.Add.sp_func,
			BO.Sub: BO.Sub.sp_func,
			BO.HadamMul: BO.Mul.sp_func,
			# BO.HadamPow: lambda exprs: sp.HadamardPower(exprs[0], exprs[1]),
			BO.Atan2: lambda _: spu.radian,
			# Vector | Vector
			BO.VecVecDot: BO.Mul.sp_func,
			BO.Cross: BO.Mul.sp_func,
			BO.VecVecOuter: BO.Mul.sp_func,
			# Matrix | Vector
			## -> A,b in Ax = b have units, and the equality must hold.
			## -> Therefore, A \ b must have the units [b]/[A].
			BO.LinSolve: lambda exprs: exprs[1] / exprs[0],
			BO.LsqSolve: lambda exprs: exprs[1] / exprs[0],
			# Vector | Matrix
			BO.VecMatOuter: BO.Mul.sp_func,
			# Matrix | Matrix
			BO.MatMatDot: BO.Mul.sp_func,
		}[self]

	@property
	def jax_func(self):
		"""Deduce an appropriate jax-based function that implements the binary operation for array inputs."""
		## TODO: Scale the units of one side to the other.
		BO = BinaryOperation

		return {
			# Number | Number
			BO.Mul: lambda exprs: exprs[0] * exprs[1],
			BO.Div: lambda exprs: exprs[0] / exprs[1],
			BO.Pow: lambda exprs: exprs[0] ** exprs[1],
			# Elements | Elements
			BO.Add: lambda exprs: exprs[0] + exprs[1],
			BO.Sub: lambda exprs: exprs[0] - exprs[1],
			BO.HadamMul: lambda exprs: exprs[0].multiply_elementwise(exprs[1]),
			BO.HadamDiv: lambda exprs: exprs[0].multiply_elementwise(
				exprs[1].applyfunc(lambda el: 1 / el)
			),
			BO.HadamPow: lambda exprs: hadamard_power(exprs[0], exprs[1]),
			BO.Atan2: lambda exprs: jnp.atan2(exprs[1], exprs[0]),
			# Vector | Vector
			BO.VecVecDot: lambda exprs: jnp.linalg.vecdot(exprs[0], exprs[1]),
			BO.Cross: lambda exprs: jnp.cross(exprs[0], exprs[1]),
			BO.VecVecOuter: lambda exprs: jnp.outer(exprs[0], exprs[1]),
			# Matrix | Vector
			BO.LinSolve: lambda exprs: jnp.linalg.solve(exprs[0], exprs[1]),
			BO.LsqSolve: lambda exprs: jnp.linalg.lstsq(exprs[0], exprs[1]),
			# Vector | Matrix
			BO.VecMatOuter: lambda exprs: jnp.outer(exprs[0], exprs[1]),
			# Matrix | Matrix
			BO.MatMatDot: lambda exprs: jnp.matmul(exprs[0], exprs[1]),
		}[self]

	####################
	# - Transforms
	####################
	def transform_funcs(self, func_l: ct.FuncFlow, func_r: ct.FuncFlow) -> ct.FuncFlow:
		"""Transform two input functions according to the current operation."""
		BO = BinaryOperation

		# Add/Sub: Normalize Unit of RHS to LHS
		## -> We can only add/sub identical units.
		## -> To be nice, we only require identical PhysicalType.
		## -> The result of a binary operation should have one unit.
		if self is BO.Add or self is BO.Sub:
			norm_func_r = func_r.scale_to_unit(func_l.func_output.unit)
		else:
			norm_func_r = func_r

		return (func_l, norm_func_r).compose_within(
			self.jax_func,
			enclosing_func_output=self.transform_outputs(
				func_l.func_output, norm_func_r.func_output
			),
			supports_jax=True,
		)

	def transform_infos(self, info_l: ct.InfoFlow, info_r: ct.InfoFlow):
		"""Deduce the output information by using `self.sp_func` to operate on the two output `SimSymbol`s, then capturing the information associated with the resulting expression.

		Warnings:
			`self` MUST be an element of `BinaryOperation.by_infos(info_l, info_r).

			If not, bad things will happen.
		"""
		return info_l.operate_output(
			info_r,
			lambda a, b: self.sp_func([a, b]),
			lambda a, b: self.unit_func([a, b]),
		)

	####################
	# - InfoFlow Transform
	####################
	def transform_outputs(
		self, output_l: sim_symbols.SimSymbol, output_r: sim_symbols.SimSymbol
	) -> sim_symbols.SimSymbol:
		# TO = TransformOperation
		return None
		# match self:
		# # Number | Number
		# case TO.Mul:
		# return
		# case TO.Div:
		# case TO.Pow:

		# # Elements | Elements
		# Add = enum.auto()
		# Sub = enum.auto()
		# HadamMul = enum.auto()
		# HadamPow = enum.auto()
		# HadamDiv = enum.auto()
		# Atan2 = enum.auto()

		# # Vector | Vector
		# VecVecDot = enum.auto()
		# Cross = enum.auto()
		# VecVecOuter = enum.auto()

		# # Matrix | Vector
		# LinSolve = enum.auto()
		# LsqSolve = enum.auto()

		# # Vector | Matrix
		# VecMatOuter = enum.auto()

		# # Matrix | Matrix
		# MatMatDot = enum.auto()
