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
from fractions import Fraction

import sympy as sp

from . import extra_sympy_units as spux

int_min = -(2**64)
int_max = 2**64
float_min = sys.float_info.min
float_max = sys.float_info.max


####################
# - Simulation Symbol Names
####################
class SimSymbolName(enum.StrEnum):
	# Lower
	LowerA = enum.auto()
	LowerB = enum.auto()
	LowerC = enum.auto()
	LowerD = enum.auto()
	LowerI = enum.auto()
	LowerT = enum.auto()
	LowerX = enum.auto()
	LowerY = enum.auto()
	LowerZ = enum.auto()

	# Fields
	Ex = enum.auto()
	Ey = enum.auto()
	Ez = enum.auto()
	Hx = enum.auto()
	Hy = enum.auto()
	Hz = enum.auto()

	Er = enum.auto()
	Etheta = enum.auto()
	Ephi = enum.auto()
	Hr = enum.auto()
	Htheta = enum.auto()
	Hphi = enum.auto()

	# Optics
	Wavelength = enum.auto()
	Frequency = enum.auto()

	Flux = enum.auto()

	PermXX = enum.auto()
	PermYY = enum.auto()
	PermZZ = enum.auto()

	DiffOrderX = enum.auto()
	DiffOrderY = enum.auto()

	# Generic
	Expr = enum.auto()

	####################
	# - UI
	####################
	@staticmethod
	def to_name(v: typ.Self) -> str:
		"""Convert the enum value to a human-friendly name.

		Notes:
			Used to print names in `EnumProperty`s based on this enum.

		Returns:
			A human-friendly name corresponding to the enum value.
		"""
		return SimSymbolName(v).name

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
	# - Computed Properties
	####################
	@property
	def name(self) -> str:
		SSN = SimSymbolName
		return {
			# Lower
			SSN.LowerA: 'a',
			SSN.LowerB: 'b',
			SSN.LowerC: 'c',
			SSN.LowerD: 'd',
			SSN.LowerI: 'i',
			SSN.LowerT: 't',
			SSN.LowerX: 'x',
			SSN.LowerY: 'y',
			SSN.LowerZ: 'z',
			# Fields
			SSN.Ex: 'Ex',
			SSN.Ey: 'Ey',
			SSN.Ez: 'Ez',
			SSN.Hx: 'Hx',
			SSN.Hy: 'Hy',
			SSN.Hz: 'Hz',
			SSN.Er: 'Ex',
			SSN.Etheta: 'Ey',
			SSN.Ephi: 'Ez',
			SSN.Hr: 'Hx',
			SSN.Htheta: 'Hy',
			SSN.Hphi: 'Hz',
			# Optics
			SSN.Wavelength: 'wl',
			SSN.Frequency: 'freq',
			SSN.Flux: 'flux',
			SSN.PermXX: 'eps_xx',
			SSN.PermYY: 'eps_yy',
			SSN.PermZZ: 'eps_zz',
			SSN.DiffOrderX: 'order_x',
			SSN.DiffOrderY: 'order_y',
			# Generic
			SSN.Expr: 'expr',
		}[self]

	@property
	def name_pretty(self) -> str:
		SSN = SimSymbolName
		return {
			SSN.Wavelength: 'λ',
			SSN.Frequency: '𝑓',
		}.get(self, self.name)


####################
# - Simulation Symbol
####################
def mk_interval(
	interval_finite: tuple[int | Fraction | float, int | Fraction | float],
	interval_inf: tuple[bool, bool],
	interval_closed: tuple[bool, bool],
	unit_factor: typ.Literal[1] | spux.Unit,
) -> sp.Interval:
	"""Create a symbolic interval from the tuples (and unit) defining it."""
	return sp.Interval(
		start=(interval_finite[0] * unit_factor if not interval_inf[0] else -sp.oo),
		end=(interval_finite[1] * unit_factor if not interval_inf[1] else sp.oo),
		left_open=(True if interval_inf[0] else not interval_closed[0]),
		right_open=(True if interval_inf[1] else not interval_closed[1]),
	)


@dataclasses.dataclass(kw_only=True, frozen=True)
class SimSymbol:
	"""A declarative representation of a symbolic variable.

	`sympy`'s symbols aren't quite flexible enough for our needs: The symbols that we're transporting often need exact domain information, an associated unit dimension, and a great deal of determinism in checks thereof.

	This dataclass is UI-friendly, as it only uses field type annotations/defaults supported by `bl_cache.BLProp`.
	It's easy to persist, easy to transport, and has many helpful properties which greatly simplify working with symbols.
	"""

	sym_name: SimSymbolName
	mathtype: spux.MathType = spux.MathType.Real
	physical_type: spux.PhysicalType = spux.PhysicalType.NonPhysical

	# Units
	## -> 'None' indicates that no particular unit has yet been chosen.
	## -> Not exposed in the UI; must be set some other way.
	unit: spux.Unit | None = None

	# Size
	## -> All SimSymbol sizes are "2D", but interpreted by convention.
	## -> 1x1: "Scalar".
	## -> nx1: "Vector".
	## -> 1xn: "Covector".
	## -> nxn: "Matrix".
	rows: int = 1
	cols: int = 1

	# Scalar Domain: "Interval"
	## -> NOTE: interval_finite_*[0] must be strictly smaller than [1].
	## -> See self.domain.
	## -> We have to deconstruct symbolic interval semantics a bit for UI.
	interval_finite_z: tuple[int, int] = (0, 1)
	interval_finite_q: tuple[tuple[int, int], tuple[int, int]] = ((0, 1), (1, 1))
	interval_finite_re: tuple[float, float] = (0, 1)
	interval_inf: tuple[bool, bool] = (True, True)
	interval_closed: tuple[bool, bool] = (False, False)

	interval_finite_im: tuple[float, float] = (0, 1)
	interval_inf_im: tuple[bool, bool] = (True, True)
	interval_closed_im: tuple[bool, bool] = (False, False)

	####################
	# - Properties
	####################
	@property
	def name(self) -> str:
		"""Usable name for the symbol."""
		return self.sym_name.name

	@property
	def name_pretty(self) -> str:
		"""Pretty (possibly unicode) name for the thing."""
		return self.sym_name.name_pretty
		## TODO: Formatting conventions for bolding/etc. of vectors/mats/...

	@property
	def plot_label(self) -> str:
		"""Pretty plot-oriented label."""
		return f'{self.name_pretty}' + (
			f'({self.unit})' if self.unit is not None else ''
		)

	@property
	def unit_factor(self) -> spux.SympyExpr:
		"""Factor corresponding to the tracked unit, which can be multiplied onto exported values without `None`-checking."""
		return self.unit if self.unit is not None else sp.S(1)

	@property
	def shape(self) -> tuple[int, ...]:
		match (self.rows, self.cols):
			case (1, 1):
				return ()
			case (_, 1):
				return (self.rows,)
			case (1, _):
				return (1, self.rows)
			case (_, _):
				return (self.rows, self.cols)

	@property
	def domain(self) -> sp.Interval | sp.Set:
		"""Return the scalar domain of valid values for each element of the symbol.

		For integer/rational/real symbols, the domain is an interval defined using the `interval_*` properties.
		This interval **must** have the property`start <= stop`.

		Otherwise, the domain is the symbolic set corresponding to `self.mathtype`.
		"""
		match self.mathtype:
			case spux.MathType.Integer:
				return mk_interval(
					self.interval_finite_z,
					self.interval_inf,
					self.interval_closed,
					self.unit_factor,
				)

			case spux.MathType.Rational:
				return mk_interval(
					Fraction(*self.interval_finite_q),
					self.interval_inf,
					self.interval_closed,
					self.unit_factor,
				)

			case spux.MathType.Real:
				return mk_interval(
					self.interval_finite_re,
					self.interval_inf,
					self.interval_closed,
					self.unit_factor,
				)

			case spux.MathType.Complex:
				return (
					mk_interval(
						self.interval_finite_re,
						self.interval_inf,
						self.interval_closed,
						self.unit_factor,
					),
					mk_interval(
						self.interval_finite_im,
						self.interval_inf_im,
						self.interval_closed_im,
						self.unit_factor,
					),
				)

	####################
	# - Properties
	####################
	@property
	def sp_symbol(self) -> sp.Symbol:
		"""Return a symbolic variable w/unit, corresponding to this `SimSymbol`.

		As much as possible, appropriate `assumptions` are set in the constructor of `sp.Symbol`, insofar as they can be determined.

		- **MathType**: Depending on `self.mathtype`.
		- **Positive/Negative**: Depending on `self.domain`.
		- **Nonzero**: Depending on `self.domain`, including open/closed boundary specifications.

		Notes:
			**The assumptions system is rather limited**, and implementations should strongly consider transporting `SimSymbols` instead of `sp.Symbol`.

			This allows tracking ex. the valid interval domain for a symbol.
		"""
		# MathType Assumption
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

		# Non-Zero Assumption
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

		# Positive/Negative Assumption
		if self.domain.left >= 0:
			mathtype_kwargs |= {'positive': True}
		elif self.domain.right <= 0:
			mathtype_kwargs |= {'negative': True}

		return sp.Symbol(self.sym_name.name, **mathtype_kwargs) * self.unit_factor

	####################
	# - Operations
	####################
	def update(self, **kwargs) -> typ.Self:
		def get_attr(attr: str):
			_notfound = 'notfound'
			if kwargs.get(attr, _notfound) is _notfound:
				return getattr(self, attr)
			return kwargs[attr]

		return SimSymbol(
			sym_name=get_attr('sym_name'),
			mathtype=get_attr('mathtype'),
			physical_type=get_attr('physical_type'),
			unit=get_attr('unit'),
			rows=get_attr('rows'),
			cols=get_attr('cols'),
			interval_finite_z=get_attr('interval_finite_z'),
			interval_finite_q=get_attr('interval_finite_q'),
			interval_finite_re=get_attr('interval_finite_q'),
			interval_inf=get_attr('interval_inf'),
			interval_closed=get_attr('interval_closed'),
			interval_finite_im=get_attr('interval_finite_im'),
			interval_inf_im=get_attr('interval_inf_im'),
			interval_closed_im=get_attr('interval_closed_im'),
		)

	def set_size(self, rows: int, cols: int) -> typ.Self:
		return SimSymbol(
			sym_name=self.sym_name,
			mathtype=self.mathtype,
			physical_type=self.physical_type,
			unit=self.unit,
			rows=rows,
			cols=cols,
			interval_finite_z=self.interval_finite_z,
			interval_finite_q=self.interval_finite_q,
			interval_finite_re=self.interval_finite_re,
			interval_inf=self.interval_inf,
			interval_closed=self.interval_closed,
			interval_finite_im=self.interval_finite_im,
			interval_inf_im=self.interval_inf_im,
			interval_closed_im=self.interval_closed_im,
		)


####################
# - Common Sim Symbols
####################
class CommonSimSymbol(enum.StrEnum):
	"""Identifiers for commonly used `SimSymbol`s, with all information about ex. `MathType`, `PhysicalType`, and (in general) valid intervals all pre-loaded.

	The enum is UI-compatible making it easy to declare a UI-driven dropdown of commonly used symbols that will all behave as expected.

	Attributes:
		X:
		Time: A symbol representing a real-valued wavelength.
		Wavelength: A symbol representing a real-valued wavelength.
			Implicitly, this symbol often represents "vacuum wavelength" in particular.
		Wavelength: A symbol representing a real-valued frequency.
			Generally, this is the non-angular frequency.
	"""

	Index = enum.auto()

	# Space|Time
	SpaceX = enum.auto()
	SpaceY = enum.auto()
	SpaceZ = enum.auto()

	AngR = enum.auto()
	AngTheta = enum.auto()
	AngPhi = enum.auto()

	DirX = enum.auto()
	DirY = enum.auto()
	DirZ = enum.auto()

	Time = enum.auto()

	# Fields
	FieldEx = enum.auto()
	FieldEy = enum.auto()
	FieldEz = enum.auto()
	FieldHx = enum.auto()
	FieldHy = enum.auto()
	FieldHz = enum.auto()

	FieldEr = enum.auto()
	FieldEtheta = enum.auto()
	FieldEphi = enum.auto()
	FieldHr = enum.auto()
	FieldHtheta = enum.auto()
	FieldHphi = enum.auto()

	# Optics
	Wavelength = enum.auto()
	Frequency = enum.auto()

	DiffOrderX = enum.auto()
	DiffOrderY = enum.auto()

	Flux = enum.auto()

	WaveVecX = enum.auto()
	WaveVecY = enum.auto()
	WaveVecZ = enum.auto()

	####################
	# - UI
	####################
	@staticmethod
	def to_name(v: typ.Self) -> str:
		"""Convert the enum value to a human-friendly name.

		Notes:
			Used to print names in `EnumProperty`s based on this enum.

		Returns:
			A human-friendly name corresponding to the enum value.
		"""
		return CommonSimSymbol(v).name

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
		SSN = SimSymbolName
		CSS = CommonSimSymbol
		return {
			CSS.Index: SSN.LowerI,
			# Space|Time
			CSS.SpaceX: SSN.LowerX,
			CSS.SpaceY: SSN.LowerY,
			CSS.SpaceZ: SSN.LowerZ,
			CSS.AngR: SSN.LowerR,
			CSS.AngTheta: SSN.LowerTheta,
			CSS.AngPhi: SSN.LowerPhi,
			CSS.DirX: SSN.LowerX,
			CSS.DirY: SSN.LowerY,
			CSS.DirZ: SSN.LowerZ,
			CSS.Time: SSN.LowerT,
			# Fields
			CSS.FieldEx: SSN.Ex,
			CSS.FieldEy: SSN.Ey,
			CSS.FieldEz: SSN.Ez,
			CSS.FieldHx: SSN.Hx,
			CSS.FieldHy: SSN.Hy,
			CSS.FieldHz: SSN.Hz,
			CSS.FieldEr: SSN.Er,
			CSS.FieldHr: SSN.Hr,
			# Optics
			CSS.Frequency: SSN.Frequency,
			CSS.Wavelength: SSN.Wavelength,
			CSS.DiffOrderX: SSN.DiffOrderX,
			CSS.DiffOrderY: SSN.DiffOrderY,
		}[self]

	def sim_symbol(self, unit: spux.Unit | None) -> SimSymbol:
		"""Retrieve the `SimSymbol` associated with the `CommonSimSymbol`."""
		CSS = CommonSimSymbol

		# Space
		sym_space = SimSymbol(
			sym_name=self.name,
			physical_type=spux.PhysicalType.Length,
			unit=unit,
		)
		sym_ang = SimSymbol(
			sym_name=self.name,
			physical_type=spux.PhysicalType.Angle,
			unit=unit,
		)

		# Fields
		def sym_field(eh: typ.Literal['e', 'h']) -> SimSymbol:
			return SimSymbol(
				sym_name=self.name,
				physical_type=spux.PhysicalType.EField
				if eh == 'e'
				else spux.PhysicalType.HField,
				unit=unit,
				interval_finite_re=(0, sys.float_info.max),
				interval_inf_re=(False, True),
				interval_closed_re=(True, False),
				interval_finite_im=(sys.float_info.min, sys.float_info.max),
				interval_inf_im=(True, True),
			)

		return {
			CSS.Index: SimSymbol(
				sym_name=self.name,
				mathtype=spux.MathType.Integer,
				interval_finite_z=(0, 2**64),
				interval_inf=(False, True),
				interval_closed=(True, False),
			),
			# Space|Time
			CSS.SpaceX: sym_space,
			CSS.SpaceY: sym_space,
			CSS.SpaceZ: sym_space,
			CSS.AngR: sym_space,
			CSS.AngTheta: sym_ang,
			CSS.AngPhi: sym_ang,
			CSS.Time: SimSymbol(
				sym_name=self.name,
				physical_type=spux.PhysicalType.Time,
				unit=unit,
				interval_finite_re=(0, sys.float_info.max),
				interval_inf=(False, True),
				interval_closed=(True, False),
			),
			# Fields
			CSS.FieldEx: sym_field('e'),
			CSS.FieldEy: sym_field('e'),
			CSS.FieldEz: sym_field('e'),
			CSS.FieldHx: sym_field('h'),
			CSS.FieldHy: sym_field('h'),
			CSS.FieldHz: sym_field('h'),
			CSS.FieldEr: sym_field('e'),
			CSS.FieldEtheta: sym_field('e'),
			CSS.FieldEphi: sym_field('e'),
			CSS.FieldHr: sym_field('h'),
			CSS.FieldHtheta: sym_field('h'),
			CSS.FieldHphi: sym_field('h'),
			CSS.Flux: SimSymbol(
				sym_name=SimSymbolName.Flux,
				mathtype=spux.MathType.Real,
				physical_type=spux.PhysicalType.Power,
				unit=unit,
			),
			# Optics
			CSS.Wavelength: SimSymbol(
				sym_name=self.name,
				mathtype=spux.MathType.Real,
				physical_type=spux.PhysicalType.Length,
				unit=unit,
				interval_finite=(0, sys.float_info.max),
				interval_inf=(False, True),
				interval_closed=(False, False),
			),
			CSS.Frequency: SimSymbol(
				sym_name=self.name,
				mathtype=spux.MathType.Real,
				physical_type=spux.PhysicalType.Freq,
				unit=unit,
				interval_finite=(0, sys.float_info.max),
				interval_inf=(False, True),
				interval_closed=(False, False),
			),
		}[self]


####################
# - Selected Direct-Access to SimSymbols
####################
idx = CommonSimSymbol.Index.sim_symbol
t = CommonSimSymbol.Time.sim_symbol
wl = CommonSimSymbol.Wavelength.sim_symbol
freq = CommonSimSymbol.Frequency.sim_symbol

space_x = CommonSimSymbol.SpaceX.sim_symbol
space_y = CommonSimSymbol.SpaceY.sim_symbol
space_z = CommonSimSymbol.SpaceZ.sim_symbol

dir_x = CommonSimSymbol.DirX.sim_symbol
dir_y = CommonSimSymbol.DirY.sim_symbol
dir_z = CommonSimSymbol.DirZ.sim_symbol

ang_r = CommonSimSymbol.AngR.sim_symbol
ang_theta = CommonSimSymbol.AngTheta.sim_symbol
ang_phi = CommonSimSymbol.AngPhi.sim_symbol

field_ex = CommonSimSymbol.FieldEx.sim_symbol
field_ey = CommonSimSymbol.FieldEy.sim_symbol
field_ez = CommonSimSymbol.FieldEz.sim_symbol
field_hx = CommonSimSymbol.FieldHx.sim_symbol
field_hy = CommonSimSymbol.FieldHx.sim_symbol
field_hz = CommonSimSymbol.FieldHx.sim_symbol

flux = CommonSimSymbol.Flux.sim_symbol

diff_order_x = CommonSimSymbol.DiffOrderX.sim_symbol
diff_order_y = CommonSimSymbol.DiffOrderY.sim_symbol
