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


class ReduceOperation(enum.StrEnum):
	# Summary
	Count = enum.auto()

	# Statistics
	Mean = enum.auto()
	Std = enum.auto()
	Var = enum.auto()

	StdErr = enum.auto()

	Min = enum.auto()
	Q25 = enum.auto()
	Median = enum.auto()
	Q75 = enum.auto()
	Max = enum.auto()

	Mode = enum.auto()

	# Reductions
	Sum = enum.auto()
	Prod = enum.auto()

	####################
	# - UI
	####################
	@staticmethod
	def to_name(value: typ.Self) -> str:
		"""A human-readable UI-oriented name for a physical type."""
		RO = ReduceOperation
		return {
			# Summary
			RO.Count: '# [a]',
			RO.Mode: 'mode [a]',
			# Statistics
			RO.Mean: 'μ [a]',
			RO.Std: 'σ [a]',
			RO.Var: 'σ² [a]',
			RO.StdErr: 'stderr [a]',
			RO.Min: 'min [a]',
			RO.Q25: 'q₂₅ [a]',
			RO.Median: 'median [a]',
			RO.Q75: 'q₇₅ [a]',
			RO.Min: 'max [a]',
			# Reductions
			RO.Sum: 'sum [a]',
			RO.Prod: 'prod [a]',
		}[value]

	@staticmethod
	def to_icon(_: typ.Self) -> str:
		"""No icons."""
		return ''

	def bl_enum_element(self, i: int) -> ct.BLEnumElement:
		"""Given an integer index, generate an element that conforms to the requirements of `bpy.props.EnumProperty.items`."""
		RO = ReduceOperation
		return (
			str(self),
			RO.to_name(self),
			RO.to_name(self),
			RO.to_icon(self),
			i,
		)

	####################
	# - Derivation
	####################
	@staticmethod
	def from_info(info: ct.InfoFlow) -> list[typ.Self]:
		"""Derive valid reduction operations from the `InfoFlow` of the operand."""
		pass

	####################
	# - Composable Functions
	####################
	@property
	def jax_func(self):
		RO = ReduceOperation
		return {}[self]

	####################
	# - Transforms
	####################
	def transform_info(self, info: ct.InfoFlow):
		pass
