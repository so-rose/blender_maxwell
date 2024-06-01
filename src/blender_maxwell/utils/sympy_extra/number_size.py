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

import sympy as sp

from blender_maxwell import contracts as ct


####################
# - Size: 1D
####################
class NumberSize1D(enum.StrEnum):
	"""Valid 1D-constrained shape."""

	Scalar = enum.auto()
	Vec2 = enum.auto()
	Vec3 = enum.auto()
	Vec4 = enum.auto()

	@staticmethod
	def to_name(value: typ.Self) -> str:
		NS = NumberSize1D
		return {
			NS.Scalar: 'Scalar',
			NS.Vec2: '2D',
			NS.Vec3: '3D',
			NS.Vec4: '4D',
		}[value]

	@staticmethod
	def to_icon(value: typ.Self) -> str:
		NS = NumberSize1D
		return {
			NS.Scalar: '',
			NS.Vec2: '',
			NS.Vec3: '',
			NS.Vec4: '',
		}[value]

	def bl_enum_element(self, i: int) -> ct.BLEnumElement:
		return (
			str(self),
			NumberSize1D.to_name(self),
			NumberSize1D.to_name(self),
			NumberSize1D.to_icon(self),
			i,
		)

	@staticmethod
	def has_shape(shape: tuple[int, ...] | None):
		return shape in [None, (2,), (3,), (4,), (2, 1), (3, 1), (4, 1)]

	def supports_shape(self, shape: tuple[int, ...] | None):
		NS = NumberSize1D
		match self:
			case NS.Scalar:
				return shape is None
			case NS.Vec2:
				return shape in ((2,), (2, 1))
			case NS.Vec3:
				return shape in ((3,), (3, 1))
			case NS.Vec4:
				return shape in ((4,), (4, 1))

	@staticmethod
	def from_shape(shape: tuple[typ.Literal[2, 3]] | None) -> typ.Self:
		NS = NumberSize1D
		return {
			None: NS.Scalar,
			(2,): NS.Vec2,
			(3,): NS.Vec3,
			(4,): NS.Vec4,
			(2, 1): NS.Vec2,
			(3, 1): NS.Vec3,
			(4, 1): NS.Vec4,
		}[shape]

	@property
	def rows(self):
		NS = NumberSize1D
		return {
			NS.Scalar: 1,
			NS.Vec2: 2,
			NS.Vec3: 3,
			NS.Vec4: 4,
		}[self]

	@property
	def cols(self):
		return 1

	@property
	def shape(self):
		NS = NumberSize1D
		return {
			NS.Scalar: None,
			NS.Vec2: (2,),
			NS.Vec3: (3,),
			NS.Vec4: (4,),
		}[self]


def symbol_range(sym: sp.Symbol) -> str:
	return f'{sym.name} ∈ ' + (
		'ℂ'
		if sym.is_complex
		else ('ℝ' if sym.is_real else ('ℤ' if sym.is_integer else '?'))
	)


####################
# - Symbol Sizes
####################
class NumberSize2D(enum.StrEnum):
	"""Simple subset of sizes for rank-2 tensors."""

	Scalar = enum.auto()

	# Vectors
	Vec2 = enum.auto()  ## 2x1
	Vec3 = enum.auto()  ## 3x1
	Vec4 = enum.auto()  ## 4x1

	# Covectors
	CoVec2 = enum.auto()  ## 1x2
	CoVec3 = enum.auto()  ## 1x3
	CoVec4 = enum.auto()  ## 1x4

	# Square Matrices
	Mat22 = enum.auto()  ## 2x2
	Mat33 = enum.auto()  ## 3x3
	Mat44 = enum.auto()  ## 4x4
