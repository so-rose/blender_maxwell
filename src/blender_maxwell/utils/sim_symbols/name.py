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
import string
import typing as typ

from blender_maxwell.utils import logger

log = logger.get(__name__)

####################
# - Simulation Symbol Names
####################
_l = ''
_it_lower = iter(string.ascii_lowercase)


class SimSymbolName(enum.StrEnum):
	# Generic
	Constant = enum.auto()
	Expr = enum.auto()
	Data = enum.auto()

	# Ascii Letters
	while True:
		try:
			globals()['_l'] = next(globals()['_it_lower'])
		except StopIteration:
			break

		locals()[f'Lower{globals()["_l"].upper()}'] = enum.auto()
		locals()[f'Upper{globals()["_l"].upper()}'] = enum.auto()

	# Greek Letters
	LowerTheta = enum.auto()
	LowerPhi = enum.auto()

	# EM Fields
	FieldE = enum.auto()
	FieldH = enum.auto()
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

	Perm = enum.auto()
	PermXX = enum.auto()
	PermYY = enum.auto()
	PermZZ = enum.auto()

	Flux = enum.auto()

	DiffOrderX = enum.auto()
	DiffOrderY = enum.auto()

	BlochX = enum.auto()
	BlochY = enum.auto()
	BlochZ = enum.auto()

	# New Backwards Compatible Entries
	## -> Ordered lists carry a particular enum integer index.
	## -> Therefore, anything but adding an index breaks backwards compat.
	## -> ...With all previous files.
	ConstantRange = enum.auto()
	Count = enum.auto()

	RelEpsRe = enum.auto()
	RelEpsIm = enum.auto()

	SimAxisIdx = enum.auto()

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
		return (
			# Ascii Letters
			{SSN[f'Lower{letter.upper()}']: letter for letter in string.ascii_lowercase}
			| {
				SSN[f'Upper{letter.upper()}']: letter.upper()
				for letter in string.ascii_lowercase
			}
			| {
				# Generic
				SSN.Constant: 'cst',
				SSN.ConstantRange: 'cst_range',
				SSN.Expr: 'expr',
				SSN.Data: 'data',
				SSN.Count: 'count',
				# Greek Letters
				SSN.LowerTheta: 'theta',
				SSN.LowerPhi: 'phi',
				# Fields
				SSN.FieldE: 'E*',
				SSN.FieldH: 'H*',
				SSN.Ex: 'Ex',
				SSN.Ey: 'Ey',
				SSN.Ez: 'Ez',
				SSN.Hx: 'Hx',
				SSN.Hy: 'Hy',
				SSN.Hz: 'Hz',
				SSN.Er: 'Er',
				SSN.Etheta: 'Ey',
				SSN.Ephi: 'Ez',
				SSN.Hr: 'Hx',
				SSN.Htheta: 'Hy',
				SSN.Hphi: 'Hz',
				# Optics
				SSN.Wavelength: 'wl',
				SSN.Frequency: 'freq',
				SSN.Perm: 'eps_r',
				SSN.PermXX: 'eps_xx',
				SSN.PermYY: 'eps_yy',
				SSN.PermZZ: 'eps_zz',
				SSN.Flux: 'flux',
				SSN.DiffOrderX: 'order_x',
				SSN.DiffOrderY: 'order_y',
				SSN.BlochX: 'bloch_x',
				SSN.BlochY: 'bloch_y',
				SSN.BlochZ: 'bloch_z',
				SSN.RelEpsRe: 'eps_r_re',
				SSN.RelEpsIm: 'eps_r_im',
				SSN.SimAxisIdx: '[xyz]',
			}
		)[self]

	@property
	def name_pretty(self) -> str:
		SSN = SimSymbolName
		return {
			# Generic
			SSN.Count: '#',
			# Greek Letters
			SSN.LowerTheta: 'θ',
			SSN.LowerPhi: 'φ',
			# Fields
			SSN.Er: 'Er',
			SSN.Etheta: 'Eθ',
			SSN.Ephi: 'Eφ',
			SSN.Hr: 'Hr',
			SSN.Htheta: 'Hθ',
			SSN.Hphi: 'Hφ',
			# Optics
			SSN.Wavelength: 'λ',
			SSN.Frequency: 'fᵣ',
			SSN.Perm: 'εᵣ',
			SSN.PermXX: 'εᵣ[xx]',
			SSN.PermYY: 'εᵣ[yy]',
			SSN.PermZZ: 'εᵣ[zz]',
			SSN.RelEpsRe: 'ℝ[εᵣ]',
			SSN.RelEpsIm: '𝕀[εᵣ]',
		}.get(self, self.name)
