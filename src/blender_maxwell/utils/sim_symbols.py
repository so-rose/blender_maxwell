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
import sys
import typing as typ

import sympy as sp

from . import extra_sympy_units as spux


####################
# - Simulation Symbols
####################
class SimSymbolName(enum.StrEnum):
	LowerA = enum.auto()
	LowerT = enum.auto()
	LowerX = enum.auto()
	LowerY = enum.auto()
	LowerZ = enum.auto()

	# Physics
	Wavelength = enum.auto()
	Frequency = enum.auto()

	@staticmethod
	def to_name(v: typ.Self) -> str:
		"""Convert the enum value to a human-friendly name.

		Notes:
			Used to print names in `EnumProperty`s based on this enum.

		Returns:
			A human-friendly name corresponding to the enum value.
		"""
		return SimSymbolName(v).name

	@property
	def name(self) -> str:
		SSN = SimSymbolName
		return {
			SSN.LowerA: 'a',
			SSN.LowerT: 't',
			SSN.LowerX: 'x',
			SSN.LowerY: 'y',
			SSN.LowerZ: 'z',
			SSN.Wavelength: 'wl',
			SSN.Frequency: 'freq',
		}[self]

	@property
	def name_pretty(self) -> str:
		SSN = SimSymbolName
		return {
			SSN.Wavelength: 'λ',
			SSN.Frequency: '𝑓',
		}.get(self, self.name)

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
	"""A declarative representation of a symbolic variable.

	`sympy`'s symbols aren't quite flexible enough for our needs: The symbols that we're transporting often need exact domain information, an associated unit dimension, and a great deal of determinism in checks thereof.

	This dataclass is UI-friendly, as it only uses field type annotations/defaults supported by `bl_cache.BLProp`.
	It's easy to persist, easy to transport, and has many helpful properties which greatly simplify working with symbols.
	"""

	sim_node_name: SimSymbolName = SimSymbolName.LowerX
	mathtype: spux.MathType = spux.MathType.Real

	physical_type: spux.PhysicalType = spux.PhysicalType.NonPhysical

	## TODO: Shape/size support? Incl. MatrixSymbol.

	# Domain
	interval_finite: tuple[float, float] = (0, 1)
	interval_inf: tuple[bool, bool] = (True, True)
	interval_closed: tuple[bool, bool] = (False, False)

	####################
	# - Properties
	####################
	@property
	def name(self) -> str:
		return self.sim_node_name.name

	@property
	def domain(self) -> sp.Interval | sp.Set:
		"""Return the domain of valid values for the symbol.

		For integer/rational/real symbols, the domain is an interval defined using the `interval_*` properties.
		This interval **must** have the property`start <= stop`.

		Otherwise, the domain is the symbolic set corresponding to `self.mathtype`.
		"""
		if self.mathtype in [
			spux.MathType.Integer,
			spux.MathType.Rational,
			spux.MathType.Real,
		]:
			return sp.Interval(
				start=self.interval_finite[0] if not self.interval_inf[0] else -sp.oo,
				end=self.interval_finite[1] if not self.interval_inf[1] else sp.oo,
				left_open=(
					True if self.interval_inf[0] else not self.interval_closed[0]
				),
				right_open=(
					True if self.interval_inf[1] else not self.interval_closed[1]
				),
			)

		return self.mathtype.symbolic_set

	####################
	# - Properties
	####################
	@property
	def sp_symbol(self) -> sp.Symbol:
		"""Return a symbolic variable corresponding to this `SimSymbol`.

		As much as possible, appropriate `assumptions` are set in the constructor of `sp.Symbol`, insofar as they can be determined.

		However, the assumptions system alone is rather limited, and implementations should therefore also strongly consider transporting `SimSymbols` directly, instead of `sp.Symbol`.
		This allows making use of other properties like `self.domain`, when appropriate.
		"""
		# MathType Domain Constraint
		## -> We must feed the assumptions system.
		mathtype_kwargs = {}
		match self.mathtype:
			case spux.MathType.Integer:
				mathtype_kwargs |= {'integer': True}
			case spux.MathType.Rational:
				mathtype_kwargs |= {'rational': True}
			case spux.MathType.Real:
				mathtype_kwargs |= {'real': True}
			case spux.MathType.Complex:
				mathtype_kwargs |= {'complex': True}

		# Interval Constraints
		if isinstance(self.domain, sp.Interval):
			# Assumption: Non-Zero
			if (
				(
					self.domain.left == 0
					and self.domain.left_open
					or self.domain.right == 0
					and self.domain.right_open
				)
				or self.domain.left > 0
				or self.domain.right < 0
			):
				mathtype_kwargs |= {'nonzero': True}

			# Assumption: Positive/Negative
			if self.domain.left >= 0:
				mathtype_kwargs |= {'positive': True}
			elif self.domain.right <= 0:
				mathtype_kwargs |= {'negative': True}

		# Construct the Symbol
		return sp.Symbol(self.sim_node_name.name, **mathtype_kwargs)


####################
# - Common Sim Symbols
####################
class CommonSimSymbol(enum.StrEnum):
	"""A set of pre-defined symbols that might commonly be used in the context of physical simulation.

	Each entry maps directly to a particular `SimSymbol`.

	The enum is compatible with `BLField`, making it easy to declare a UI-driven dropdown of symbols that behave as expected.

	Attributes:
		Wavelength: A symbol representing a real-valued wavelength.
			Implicitly, this symbol often represents "vacuum wavelength" in particular.
		Wavelength: A symbol representing a real-valued frequency.
			Generally, this is the non-angular frequency.
	"""

	X = enum.auto()
	Time = enum.auto()
	Wavelength = enum.auto()
	Frequency = enum.auto()

	@staticmethod
	def to_name(v: typ.Self) -> str:
		"""Convert the enum value to a human-friendly name.

		Notes:
			Used to print names in `EnumProperty`s based on this enum.

		Returns:
			A human-friendly name corresponding to the enum value.
		"""
		return CommonSimSymbol(v).sim_symbol_name.name

	@staticmethod
	def to_icon(_: typ.Self) -> str:
		"""Convert the enum value to a Blender icon.

		Notes:
			Used to print icons in `EnumProperty`s based on this enum.

		Returns:
			A human-friendly name corresponding to the enum value.
		"""
		return ''

	####################
	# - Properties
	####################
	@property
	def name(self) -> str:
		return self.sim_symbol.name

	@property
	def sim_symbol_name(self) -> str:
		SSN = SimSymbolName
		CSS = CommonSimSymbol
		return {
			CSS.X: SSN.LowerX,
			CSS.Time: SSN.LowerT,
			CSS.Wavelength: SSN.Wavelength,
			CSS.Frequency: SSN.Frequency,
		}[self]

	@property
	def sim_symbol(self) -> SimSymbol:
		"""Retrieve the `SimSymbol` associated with the `CommonSimSymbol`."""
		CSS = CommonSimSymbol
		return {
			CSS.X: SimSymbol(
				sim_node_name=self.sim_symbol_name,
				mathtype=spux.MathType.Real,
				physical_type=spux.PhysicalType.NonPhysical,
				## TODO: Unit of Picosecond
				interval_finite=(sys.float_info.min, sys.float_info.max),
				interval_inf=(True, True),
				interval_closed=(False, False),
			),
			CSS.Time: SimSymbol(
				sim_node_name=self.sim_symbol_name,
				mathtype=spux.MathType.Real,
				physical_type=spux.PhysicalType.Time,
				## TODO: Unit of Picosecond
				interval_finite=(0, sys.float_info.max),
				interval_inf=(False, True),
				interval_closed=(True, False),
			),
			CSS.Wavelength: SimSymbol(
				sim_node_name=self.sim_symbol_name,
				mathtype=spux.MathType.Real,
				physical_type=spux.PhysicalType.Length,
				## TODO: Unit of Picosecond
				interval_finite=(0, sys.float_info.max),
				interval_inf=(False, True),
				interval_closed=(False, False),
			),
			CSS.Frequency: SimSymbol(
				sim_node_name=self.sim_symbol_name,
				mathtype=spux.MathType.Real,
				physical_type=spux.PhysicalType.Freq,
				interval_finite=(0, sys.float_info.max),
				interval_inf=(False, True),
				interval_closed=(False, False),
			),
		}[self]


####################
# - Selected Direct Access
####################
x = CommonSimSymbol.X.sim_symbol
t = CommonSimSymbol.Time.sim_symbol
wl = CommonSimSymbol.Wavelength.sim_symbol
freq = CommonSimSymbol.Frequency.sim_symbol
