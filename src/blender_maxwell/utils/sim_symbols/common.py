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

from blender_maxwell.utils import logger
from blender_maxwell.utils import sympy_extra as spux

from .name import SimSymbolName
from .sim_symbol import SimSymbol

log = logger.get(__name__)


####################
# - Common Sim Symbols
####################
class CommonSimSymbol(enum.StrEnum):
	"""Identifiers for commonly used `SimSymbol`s, with all information about ex. `MathType`, `PhysicalType`, and (in general) valid intervals all pre-loaded.

	The enum is UI-compatible making it easy to declare a UI-driven dropdown of commonly used symbols that will all behave as expected.

	Attributes:
		Time: A symbol representing a real-valued wavelength.
		Wavelength: A symbol representing a real-valued wavelength.
			Implicitly, this symbol often represents "vacuum wavelength" in particular.
		Wavelength: A symbol representing a real-valued frequency.
			Generally, this is the non-angular frequency.
	"""

	Index = enum.auto()
	SimAxisIdx = enum.auto()

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
	FieldE = enum.auto()
	FieldH = enum.auto()
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

	Flux = enum.auto()

	DiffOrderX = enum.auto()
	DiffOrderY = enum.auto()

	RelEpsRe = enum.auto()
	RelEpsIm = enum.auto()

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
			CSS.SimAxisIdx: SSN.SimAxisIdx,
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
			CSS.FieldE: SSN.FieldE,
			CSS.FieldH: SSN.FieldH,
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
			CSS.Flux: SSN.Flux,
			CSS.DiffOrderX: SSN.DiffOrderX,
			CSS.DiffOrderY: SSN.DiffOrderY,
			CSS.RelEpsRe: SSN.RelEpsRe,
			CSS.RelEpsIm: SSN.RelEpsIm,
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
				mathtype=spux.MathType.Complex,
				physical_type=(
					spux.PhysicalType.EField if eh == 'e' else spux.PhysicalType.HField
				),
				unit=unit,
				domain=spux.BlessedSet(
					sp.ComplexRegion(sp.Interval(0, sp.oo) * sp.Reals)
				),
			)

		return {
			CSS.Index: SimSymbol(
				sym_name=self.name,
				mathtype=spux.MathType.Integer,
				domain=spux.BlessedSet(sp.Naturals0),
			),
			CSS.SimAxisIdx: SimSymbol(
				sym_name=self.name,
				mathtype=spux.MathType.Integer,
				domain=spux.BlessedSet(sp.FiniteSet(0, 1, 2)),
			),
			# Space|Time
			CSS.SpaceX: sym_space,
			CSS.SpaceY: sym_space,
			CSS.SpaceZ: sym_space,
			CSS.AngR: sym_space,
			CSS.AngTheta: sym_ang,
			CSS.AngPhi: sym_ang,
			CSS.DirX: SimSymbol(
				sym_name=self.name,
				mathtype=spux.MathType.Real,
				physical_type=spux.PhysicalType.Length,
				unit=unit,
				domain=spux.BlessedSet(sp.Interval(-sp.oo, sp.oo)),
			),
			CSS.DirY: SimSymbol(
				sym_name=self.name,
				mathtype=spux.MathType.Real,
				physical_type=spux.PhysicalType.Length,
				unit=unit,
				domain=spux.BlessedSet(sp.Interval(-sp.oo, sp.oo)),
			),
			CSS.Time: SimSymbol(
				sym_name=self.name,
				physical_type=spux.PhysicalType.Time,
				unit=unit,
				domain=spux.BlessedSet(sp.Interval(0, sp.oo)),
			),
			# Fields
			CSS.FieldE: sym_field('e'),
			CSS.FieldH: sym_field('h'),
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
			# Optics
			CSS.Wavelength: SimSymbol(
				sym_name=self.name,
				mathtype=spux.MathType.Real,
				physical_type=spux.PhysicalType.Length,
				unit=unit,
				domain=spux.BlessedSet(sp.Interval.open(0, sp.oo)),
			),
			CSS.Frequency: SimSymbol(
				sym_name=self.name,
				mathtype=spux.MathType.Real,
				physical_type=spux.PhysicalType.Freq,
				unit=unit,
				domain=spux.BlessedSet(sp.Interval.open(0, sp.oo)),
			),
			CSS.Flux: SimSymbol(
				sym_name=SimSymbolName.Flux,
				mathtype=spux.MathType.Real,
				physical_type=spux.PhysicalType.Power,
				unit=unit,
				domain=spux.BlessedSet(sp.Interval.open(0, sp.oo)),
			),
			CSS.DiffOrderX: SimSymbol(
				sym_name=self.name,
				mathtype=spux.MathType.Integer,
				domain=spux.BlessedSet(sp.Integers),
			),
			CSS.DiffOrderY: SimSymbol(
				sym_name=self.name,
				mathtype=spux.MathType.Integer,
				domain=spux.BlessedSet(sp.Integers),
			),
			CSS.RelEpsRe: SimSymbol(
				sym_name=self.name,
				mathtype=spux.MathType.Real,
				domain=spux.BlessedSet(sp.Reals),
			),
			CSS.RelEpsIm: SimSymbol(
				sym_name=self.name,
				mathtype=spux.MathType.Real,
				domain=spux.BlessedSet(sp.Reals),
			),
		}[self]


####################
# - Selected Direct-Access to SimSymbols
####################
idx = CommonSimSymbol.Index.sim_symbol
sim_axis_idx = CommonSimSymbol.SimAxisIdx.sim_symbol
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

field_e = CommonSimSymbol.FieldE.sim_symbol
field_h = CommonSimSymbol.FieldH.sim_symbol
field_ex = CommonSimSymbol.FieldEx.sim_symbol
field_ey = CommonSimSymbol.FieldEy.sim_symbol
field_ez = CommonSimSymbol.FieldEz.sim_symbol
field_hx = CommonSimSymbol.FieldHx.sim_symbol
field_hy = CommonSimSymbol.FieldHx.sim_symbol
field_hz = CommonSimSymbol.FieldHx.sim_symbol

flux = CommonSimSymbol.Flux.sim_symbol

diff_order_x = CommonSimSymbol.DiffOrderX.sim_symbol
diff_order_y = CommonSimSymbol.DiffOrderY.sim_symbol

rel_eps_re = CommonSimSymbol.RelEpsRe.sim_symbol
rel_eps_im = CommonSimSymbol.RelEpsIm.sim_symbol
