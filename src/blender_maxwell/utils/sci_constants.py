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

"""Access `scipy.constants` using `sympy` units.

Notes:
	See <https://docs.scipy.org/doc/scipy/reference/constants.html#scipy.constants.physical_constants>

Attributes:
	_SCIPY_COMBINED_UNITS: A mapping from scipy constant names to its combined unit, represented as a tuple of "base" units.
	SCIPY_UNIT_TO_SYMPY_UNIT: A mapping from scipy constant names to its combined unit, represented as a tuple of "base" units.
	SCI_CONSTANTS: A mapping from scipy constant names to an equivalent sympy expression.expression.
	SCI_CONSTANTS_REF: Original source of constant data / choices.
"""

import functools

import scipy as sc
import sympy as sp
import sympy.physics.units as spu

from . import sympy_extra as spux

SUPPORTED_SCIPY_PREFIX = '1.12'
if not sc.version.full_version.startswith(SUPPORTED_SCIPY_PREFIX):
	msg = f'The active scipy version "{sc.version.full_version}" has not been tested with the "sci_constants.py" module, which only supports scipy version(s prefixed with) "{SUPPORTED_SCIPY_PREFIX}". While the module may otherwise work, constants available in other versions of scipy may not conform to hard-coded unit lookups; as such, we throw an error, to ensure the absence of unfortunate resulting bugs'
	raise RuntimeError(msg)

####################
# - Fundamental Constants
####################
vac_speed_of_light = sp.nsimplify(sc.constants.speed_of_light) * spu.meter / spu.second
hartree_energy = (
	sp.nsimplify(sc.constants.physical_constants['Hartree energy in eV'][0])
	* spu.electronvolt
)

####################
# - Parse scipy.constants
####################
_SCIPY_EXCLUDED_ENTRIES: set[str] = {
	'Faraday constant for conventional electric current',
}
_SCIPY_COMBINED_UNITS: dict[str, tuple[str]] = {
	name: tuple(cst[1].split(' '))
	for name, cst in sc.constants.physical_constants.items()
	if cst[1] and name not in _SCIPY_EXCLUDED_ENTRIES
}

## TODO: Better interface to the official Units section? See https://docs.scipy.org/doc/scipy/reference/constants.html#units
SCIPY_UNIT_TO_SYMPY_UNIT = {
	#'C_90': spu.coulombs, ## "Conventional electric current"
	'H': spu.henry,
	'C^4': spu.coulombs**4,
	'm^4': spu.meters**4,
	'm^-1': 1 / spu.meter,
	'J^-3': 1 / spu.joules**3,
	's^-2': 1 / spu.second**2,
	'kg': spu.kilogram,
	'eV': spu.electronvolt,
	'GeV': 10**9 * spu.electronvolt,
	'S': spu.siemens,
	'(GeV/c^2)^-2': (10**9 * spu.electronvolt / vac_speed_of_light**2) ** -2,
	'GeV^-2': (10**9 * spu.electronvolt) ** -2,
	'E_h': hartree_energy,
	'Hz': spu.hertz,
	'MeV/c': (10**6 * spu.electronvolt) / vac_speed_of_light,
	'Pa': spu.pascal,
	'ohm': spu.ohms,
	'Wb': spu.weber,
	'W^-1': 1 / spu.watt,
	'u': spu.atomic_mass_constant,
	'm^-3': 1 / spu.meters**3,
	'J': spu.joules,
	'sr^-1': 1 / spu.steradian,
	'T^-1': 1 / spu.tesla,
	'C^2': spu.coulombs**2,
	'F': spu.farad,
	'MeV': 10**6 * spu.electronvolt,
	'J^-2': 1 / spu.joules**2,
	'mol^-1': 1 / spu.mole,
	'kg^-1': 1 / spu.kilogram,
	'T^-2': 1 / spu.tesla**2,
	'fm': spux.femtometer,
	'V^-1': 1 / spu.volt,
	'N': spu.newton,
	'A': spu.ampere,
	's^-1': 1 / spu.second,
	'lm': spux.lumen,
	'm^3': spu.meters**3,
	'm^2': spu.meters**2,
	'K^-1': 1 / spu.kelvin,
	'm^-2': 1 / spu.meters**2,
	'MHz': 10**6 * spu.hertz,
	'J^-1': 1 / spu.joules,
	'Hz^-1': 1 / spu.hertz,
	'm': spu.meter,
	'C^3': spu.coulombs**3,
	'K': spu.kelvin,
	'A^-2': 1 / spu.ampere**2,
	'T': spu.tesla,
	'K^-4': 1 / spu.kelvin**4,
	'W': spu.watt,
	'C': spu.coulombs,
	's': spu.second,
	'V': spu.volt,
}

####################
# - Extract Units
####################
SCI_CONSTANT_UNITS = {
	name: functools.reduce(
		lambda a, b: a * SCIPY_UNIT_TO_SYMPY_UNIT[b],
		combined_unit,
		sp.S(1),
	)
	for name, combined_unit in _SCIPY_COMBINED_UNITS.items()
}
SCI_CONSTANTS_INFO = {
	name: {
		'units': SCI_CONSTANT_UNITS[name],
		'uncertainty': sc.constants.physical_constants[name][2],
		'scipy_units': sc.constants.physical_constants[name][1],
	}
	for name, combined_unit in _SCIPY_COMBINED_UNITS.items()
}
SCI_CONSTANTS_REF = (
	'[CODATA2018]',
	'CODATA Recommended Values of the Fundamental Physical Constants 2018.',
	'https://physics.nist.gov/cuu/Constants/',
)

SCI_CONSTANTS = {
	name: sc.constants.physical_constants[name][0] * SCI_CONSTANT_UNITS[name]
	for name, combined_unit in _SCIPY_COMBINED_UNITS.items()
}
