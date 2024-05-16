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
import enum
import typing as typ

import sympy as sp

from . import extra_sympy_units as spux


class SimSymbolNames(enum.StrEnum):
	LowerA = enum.auto()
	LowerLambda = enum.auto()

	@staticmethod
	def to_name(v: typ.Self) -> str:
		"""Convert the enum value to a human-friendly name.

		Notes:
			Used to print names in `EnumProperty`s based on this enum.

		Returns:
			A human-friendly name corresponding to the enum value.
		"""
		SSN = SimSymbolNames
		return {
			SSN.LowerA: 'a',
			SSN.LowerLambda: 'λ',
		}[v]

	@staticmethod
	def to_icon(_: typ.Self) -> str:
		"""Convert the enum value to a Blender icon.

		Notes:
			Used to print icons in `EnumProperty`s based on this enum.

		Returns:
			A human-friendly name corresponding to the enum value.
		"""
		return ''


@dataclasses.dataclass(kw_only=True, frozen=True)
class SimSymbol:
	name: SimSymbolNames = SimSymbolNames.LowerLambda
	mathtype: spux.MathType = spux.MathType.Real

	## TODO:
	## -> Physical Type: Track unit dimension information on the side.
	## -> Domain: Ability to constrain mathtype ex. (-pi,pi]
	## -> Shape: For using sp.MatrixSymbol w/predefined rows/cols.

	@property
	def sp_symbol(self):
		mathtype_kwarg = {}
		match self.mathtype:
			case spux.MathType.Real:
				mathtype_kwarg = {}

		return sp.Symbol(self.name, **mathtype_kwarg)
