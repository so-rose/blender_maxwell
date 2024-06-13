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

"""Implements transform operations for the `MapNode`."""

import enum
import functools
import typing as typ

import jax.numpy as jnp
import sympy as sp
import sympy.physics.units as spu

from blender_maxwell.utils import logger, sim_symbols
from blender_maxwell.utils import sympy_extra as spux

from .. import contracts as ct

log = logger.get(__name__)

MT = spux.MathType
PT = spux.PhysicalType


# @functools.lru_cache(maxsize=1024)
# def expand_shapes(
# shape_l: tuple[int, ...], shape_r: tuple[int, ...]
# ) -> tuple[tuple[int, ...], tuple[int, ...]]:
# """Transform each shape to two new shapes, whose lengths are identical, and for which operations between are well-defined, but which occupies the same amount of memory."""
# axes = dict(
# reversed(
# list(
# itertools.zip_longest(reversed(shape_l), reversed(shape_r), fillvalue=1)
# )
# )
# )
#
# return (tuple(axes.keys()), tuple(axes.values()))
#
#
# @functools.lru_cache(maxsize=1024)
# def broadcast_shape(
# expanded_shape_l: tuple[int, ...], expanded_shape_r: tuple[int, ...]
# ) -> tuple[int, ...] | None:
# """Deduce a common shape that an object of both expanded shapes can be broadcast to."""
# new_shape = []
# for ax_l, ax_r in itertools.zip_longest(
# expanded_shape_l, expanded_shape_r, fillvalue=1
# ):
# if ax_l == 1 or ax_r == 1 or ax_l == ax_r:  # noqa: PLR1714
# new_shape.append(max([ax_l, ax_r]))
# else:
# return None
#
# return tuple(new_shape)
#
#
# def broadcast_to_shape(
# M: sp.NDimArray, compatible_shape: tuple[int, ...]
# ) -> spux.SympyType:
# """Conform an array with expanded shape to the given shape, expanding any axes that need expanding."""
# L = M
#
# incremental_shape = ()
# for orig_ax, new_ax in reversed(zip(M.shape, compatible_shape, strict=True)):
# incremental_shape = (new_ax, *incremental_shape)
# if orig_ax == 1 and new_ax > 1:
# _L = sp.flatten(L) if L.shape == () else L.tolist()
#
# L = sp.ImmutableDenseNDimArray(_L * new_ax).reshape(*compatible_shape)
#
# return L
#
#
# def sp_operation(op, lhs: spux.SympyType, rhs: spux.SympyType) -> spux.SympyType | None:
# if not isinstance(lhs, sp.MatrixBase | sp.NDimArray) and not isinstance(
# lhs, sp.MatrixBase | sp.NDimArray
# ):
# return op(lhs, rhs)
#
# # Deduce Expanded L/R Arrays
# ## -> This conforms the shape of both operands to broadcastable shape.
# ## -> The actual memory usage from doing this remains identical.
# _L = sp.ImmutableDenseNDimArray(lhs)
# _R = sp.ImmutableDenseNDimArray(rhs)
# expanded_shape_l, expanded_shape_r = expand_shapes(_L.shape, _R.shape)
# _L = _L.reshape(*expanded_shape_l)
# _R = _R.reshape(*expanded_shape_r)
#
# # Broadcast Expanded L/R Arrays
# ## -> Expanded dimensions will be conformed to the max of each.
# ## -> This conforms the shape of both operands to broadcastable shape.
# broadcasted_shape = broadcast_to_shape(expanded_shape_l, expanded_shape_r)
# if broadcasted_shape is None:
# return None
#
# L = broadcast_to_shape(_L, broadcasted_shape)
# R = broadcast_to_shape(_R, broadcasted_shape)
#
# # Run Elementwise Operation
# ## -> Elementwise operations can now cleanly run between both operands.
# output = op(L, R)
# if output.shape in [1, 2]:
# return sp.ImmutableMatrix(output.tomatrix())
# return output
#
#
# def hadamard_product(lhs: spux.SympyType, rhs: spux.SympyType) -> spux.SympyType | None:
# match (isinstance(lhs, sp.MatrixBase), isinstance(rhs, sp.MatrixBase)):
# case (False, False):
# return lhs * rhs
#
# case (True, False):
# return lhs.applyfunc(lambda el: el * rhs)
#
# case (False, True):
# return rhs.applyfunc(lambda el: lhs * el)
#
# case (True, True) if lhs.shape == rhs.shape:
# common_shape = lhs.shape
# return sp.ImmutableMatrix(
# *common_shape, lambda i, j: lhs[i, j] ** rhs[i, j]
# )
#
# msg = f'Incompatible lhs and rhs for hadamard power: {lhs} | {rhs}'
# raise ValueError(msg)


def hadamard_power(lhs: spux.SympyType, rhs: spux.SympyType) -> spux.SympyType:
	"""Implement the Hadamard Power.

	Follows the specification in <https://docs.sympy.org/latest/modules/matrices/expressions.html#sympy.matrices.expressions.HadamardProduct>, which also conforms to `numpy` broadcasting rules for `**` on `np.ndarray`.
	"""
	match (isinstance(lhs, sp.MatrixBase), isinstance(rhs, sp.MatrixBase)):
		case (False, False):
			return lhs**rhs

		case (True, False):
			return lhs.applyfunc(lambda el: el**rhs)

		case (False, True):
			return rhs.applyfunc(lambda el: lhs**el)

		case (True, True) if lhs.shape == rhs.shape:
			common_shape = lhs.shape
			return sp.ImmutableMatrix(
				*common_shape, lambda i, j: lhs[i, j] ** rhs[i, j]
			)

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
			# Matrix | Matrix
			BO.MatMatDot: '𝐋 · 𝐑',
		}[value]

	@staticmethod
	def to_icon(_: typ.Self) -> str:
		"""No icons."""
		return ''

	@functools.cached_property
	def name(self) -> str:
		"""No icons."""
		return BinaryOperation.to_name(self)

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

	@staticmethod
	def bl_enum_elements(
		info_l: ct.InfoFlow, info_r: ct.InfoFlow
	) -> list[ct.BLEnumElement]:
		"""Generate a list of guaranteed-valid operations based on the passed `InfoFlow`s.

		Returns a `bpy.props.EnumProperty.items`-compatible list.
		"""
		return [
			operation.bl_enum_element(i)
			for i, operation in enumerate(BinaryOperation.from_infos(info_l, info_r))
		]

	####################
	# - Ops from Shape
	####################
	@staticmethod
	def from_infos(info_l: ct.InfoFlow, info_r: ct.InfoFlow) -> list[typ.Self]:  # noqa: C901, PLR0912
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
				case (ordl, ordr, _) if ordl > 0 and ordr > 0:
					## TODO: _ is not correct
					ops += [BO.HadamMul, BO.HadamDiv]

				case (ordl, ordr, False) if ordl == 0 and ordr == 0:
					ops += [BO.Mul]
				case (ordl, ordr, False) if ordl > 0 and ordr == 0:
					ops += [BO.Mul]
				case (ordl, ordr, False) if ordl == 0 and ordr > 0:
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

				case (ordl, ordr, MT.Integer) if (
					ordl > 0 and ordr == 0 and info_l.output.rows == info_l.output.cols
				):
					ops += [BO.Pow, BO.HadamPow]

				case _:
					ops += [BO.HadamPow]

		# Atan2
		if (
			(
				info_l.output.mathtype is not MT.Complex
				and info_r.output.mathtype is not MT.Complex
			)
			and (
				info_l.output.physical_type is PT.Length
				and info_r.output.physical_type is PT.Length
			)
			or (
				info_l.output.physical_type is PT.NonPhysical
				and info_l.output.unit is None
				and info_r.output.physical_type is PT.NonPhysical
				and info_r.output.unit is None
			)
		):
			ops += [BO.Atan2]

		# Operations by-Output Length
		match (
			info_l.output.shape_len,
			info_r.output.shape_len,
		):
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
				if (outl.rows == 3 and outr.rows == 3) or (  # noqa: PLR2004
					outl.cols == 3 and outl.cols == 3  # noqa: PLR2004
				):
					ops += [BO.Cross]

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

		## BODO: Make this compatible with sp.Matrix inputs
		return {
			# Number | Number
			BO.Mul: lambda exprs: exprs[0] * exprs[1],
			BO.Div: lambda exprs: exprs[0] / exprs[1],
			BO.Pow: lambda exprs: hadamard_power(exprs[0], exprs[1]),
			# Elements | Elements
			BO.Add: lambda exprs: exprs[0] + exprs[1],
			BO.Sub: lambda exprs: exprs[0] - exprs[1],
			BO.HadamMul: lambda exprs: exprs[0].multiply_elementwise(exprs[1]),
			BO.HadamDiv: lambda exprs: exprs[0].multiply_elementwise(
				exprs[1].applyfunc(lambda el: 1 / el)
			),
			BO.HadamPow: lambda exprs: hadamard_power(exprs[0], exprs[1]),
			BO.Atan2: lambda exprs: sp.atan2(exprs[1], exprs[0]),
			# Vector | Vector
			BO.VecVecDot: lambda exprs: (exprs[0].T @ exprs[1])[0],
			BO.Cross: lambda exprs: exprs[0].cross(exprs[1]),
			BO.VecVecOuter: lambda exprs: exprs[0] @ exprs[1].T,
			# Matrix | Vector
			BO.LinSolve: lambda exprs: exprs[0].solve(exprs[1]),
			BO.LsqSolve: lambda exprs: exprs[0].solve_least_squares(exprs[1]),
			# Matrix | Matrix
			BO.MatMatDot: lambda exprs: exprs[0] @ exprs[1],
		}[self]

	@property
	def scalar_sp_func(self):
		"""The binary function to apply to both unit expressions, in order to deduce the unit expression of the output."""
		BO = BinaryOperation

		## BODO: Make this compatible with sp.Matrix inputs
		return {
			# Number | Number
			BO.Mul: BO.Mul.sp_func,
			BO.Div: BO.Div.sp_func,
			BO.Pow: BO.Pow.sp_func,
			# Elements | Elements
			BO.Add: lambda exprs: exprs[0],
			BO.Sub: lambda exprs: exprs[0],
			BO.HadamMul: BO.Mul.sp_func,
			BO.HadamDiv: BO.Div.sp_func,
			BO.HadamPow: BO.Pow.sp_func,
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
			# Matrix | Matrix
			BO.MatMatDot: BO.Mul.sp_func,
		}[self]

	@property
	def jax_func(self):
		"""Deduce an appropriate jax-based function that implements the binary operation for array inputs."""
		## BODO: Scale the units of one side to the other.
		BO = BinaryOperation

		return {
			# Number | Number
			BO.Mul: lambda exprs: exprs[0] * exprs[1],
			BO.Div: lambda exprs: exprs[0] / exprs[1],
			BO.Pow: lambda exprs: exprs[0] ** exprs[1],
			# Elements | Elements
			BO.Add: lambda exprs: exprs[0] + exprs[1],
			BO.Sub: lambda exprs: exprs[0] - exprs[1],
			BO.HadamMul: lambda exprs: exprs[0] * exprs[1],
			BO.HadamDiv: lambda exprs: exprs[0] / exprs[1],
			BO.HadamPow: lambda exprs: exprs[0] ** exprs[1],
			BO.Atan2: lambda exprs: jnp.atan2(exprs[1], exprs[0]),
			# Vector | Vector
			BO.VecVecDot: lambda exprs: jnp.linalg.vecdot(exprs[0], exprs[1]),
			BO.Cross: lambda exprs: jnp.cross(exprs[0], exprs[1]),
			BO.VecVecOuter: lambda exprs: jnp.outer(exprs[0], exprs[1]),
			# Matrix | Vector
			BO.LinSolve: lambda exprs: jnp.linalg.solve(exprs[0], exprs[1]),
			BO.LsqSolve: lambda exprs: jnp.linalg.lstsq(exprs[0], exprs[1]),
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

		return (func_l | norm_func_r).compose_within(
			self.jax_func,
			enclosing_func_output=self.transform_outputs(
				func_l.func_output, func_r.func_output
			),
			supports_jax=True,
		)

	def transform_infos(self, info_l: ct.InfoFlow, info_r: ct.InfoFlow) -> ct.InfoFlow:
		"""Transform the `InfoFlow` characterizing the output."""
		if len(info_l.dims) == 0:
			dims = info_r.dims
		elif len(info_r.dims) == 0:
			dims = info_l.dims
		else:
			dims = info_l.dims

		return ct.InfoFlow(
			dims=dims,
			output=self.transform_outputs(info_l.output, info_r.output),
			pinned_values=info_l.pinned_values | info_r.pinned_values,
		)

	def transform_params(
		self, params_l: ct.ParamsFlow, params_r: ct.ParamsFlow
	) -> ct.ParamsFlow:
		"""Aggregate the incoming function parameters for the output."""
		return params_l | params_r

	####################
	# - Other Transforms
	####################
	def transform_outputs(
		self, sym_l: sim_symbols.SimSymbol, sym_r: sim_symbols.SimSymbol
	) -> sim_symbols.SimSymbol:
		BO = BinaryOperation

		if sym_l.sym_name == sym_r.sym_name:
			name = sym_l.sym_name
		else:
			name = sim_symbols.SimSymbolName.Expr

		dm_l = sym_l.domain
		dm_r = sym_r.domain
		match self:
			case BO.Mul | BO.Div | BO.Pow | BO.HadamDiv:
				# dm = self.scalar_sp_func([dm_l, dm_r])
				unit_factor = self.scalar_sp_func(
					[sym_l.unit_factor, sym_r.unit_factor]
				)
				unit = unit_factor if spux.uses_units(unit_factor) else None
				physical_type = PT.from_unit(unit, optional=True, optional_nonphy=True)

				mathtype = MT.combine(
					MT.from_symbolic_set(dm_l.bset), MT.from_symbolic_set(dm_r.bset)
				)
				return sim_symbols.SimSymbol(
					sym_name=name,
					mathtype=mathtype,
					physical_type=physical_type,
					unit=unit,
					rows=max([sym_l.rows, sym_r.rows]),
					cols=max([sym_l.cols, sym_r.cols]),
					depths=tuple(
						[
							max([dp_l, dp_r])
							for dp_l, dp_r in zip(
								sym_l.depths, sym_r.depths, strict=True
							)
						]
					),
					domain=spux.BlessedSet(mathtype.symbolic_set),
				)

			case BO.Add | BO.Sub:
				fac_r_unit_to_l_unit = sp.S(spux.scaling_factor(sym_l.unit, sym_r.unit))

				dm = self.scalar_sp_func([dm_l, dm_r * fac_r_unit_to_l_unit])
				unit_factor = self.scalar_sp_func(
					[sym_l.unit_factor, sym_r.unit_factor]
				)
				unit = unit_factor if spux.uses_units(unit_factor) else None
				physical_type = PT.from_unit(unit, optional=True, optional_nonphy=True)

				return sym_l.update(
					sym_name=name,
					mathtype=MT.from_symbolic_set(dm.bset),
					physical_type=physical_type,
					unit=None if unit_factor == 1 else unit_factor,
					domain=dm,
				)

			case BO.HadamMul | BO.HadamPow:
				# fac_r_unit_to_l_unit = sp.S(spux.scaling_factor(sym_l.unit, sym_r.unit))

				mathtype = MT.combine(
					MT.from_symbolic_set(dm_l.bset), MT.from_symbolic_set(dm_r.bset)
				)
				# dm = self.scalar_sp_func([dm_l, dm_r * fac_r_unit_to_l_unit])
				unit_factor = self.scalar_sp_func(
					[sym_l.unit_factor, sym_r.unit_factor]
				)
				unit = unit_factor if spux.uses_units(unit_factor) else None
				physical_type = PT.from_unit(unit, optional=True, optional_nonphy=True)

				return sym_l.update(
					sym_name=name,
					mathtype=mathtype,
					physical_type=physical_type,
					unit=None if unit_factor == 1 else unit_factor,
					domain=spux.BlessedSet(mathtype.symbolic_set),
				)

			case BO.Atan2:
				dm = dm_l.atan2(dm_r)

				return sym_l.update(
					sym_name=name,
					mathtype=MT.from_symbolic_set(dm.bset),
					physical_type=PT.Angle,
					unit=spu.radian,
					domain=dm,
				)

			case BO.VecVecDot:
				_dm = dm_l * dm_r
				dm = _dm + _dm

				return sym_l.update(
					sym_name=name,
					domain=dm,
					rows=1,
					cols=1,
				)

			case BO.Cross:
				_dm = dm_l * dm_r
				dm = _dm + _dm

				return sym_l.update(
					sym_name=name,
					domain=dm,
				)

			case BO.VecVecOuter:
				dm = dm_l * dm_r

				return sym_l.update(
					sym_name=name,
					domain=dm,
					rows=max([sym_l.rows, sym_r.rows]),
					cols=max([sym_l.cols, sym_r.cols]),
				)

			case BO.LinSolve | BO.LsqSolve:
				mathtype = MT.combine(
					MT.from_symbolic_set(dm_l.bset), MT.from_symbolic_set(dm_r.bset)
				).symbolic_set
				dm = spux.BlessedSet(mathtype.symbolic_set)

				return sym_r.update(mathtype=mathtype, domain=dm)

			case BO.MatMatDot:
				mathtype = MT.combine(
					MT.from_symbolic_set(dm_l.bset), MT.from_symbolic_set(dm_r.bset)
				).symbolic_set
				dm = spux.BlessedSet(mathtype.symbolic_set)

				return sym_r.update(mathtype=mathtype, domain=dm)
