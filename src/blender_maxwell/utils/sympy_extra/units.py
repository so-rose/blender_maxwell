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

import typing as typ

import sympy as sp
import sympy.physics.units as spu

####################
# - Units
####################
# Time
femtosecond = fs = spu.Quantity('femtosecond', abbrev='fs')
femtosecond.set_global_relative_scale_factor(spu.femto, spu.second)

# Length
femtometer = fm = spu.Quantity('femtometer', abbrev='fm')
femtometer.set_global_relative_scale_factor(spu.femto, spu.meter)

# Lum Flux
lumen = lm = spu.Quantity('lumen', abbrev='lm')
lumen.set_global_relative_scale_factor(1, spu.candela * spu.steradian)

# Force
nanonewton = nN = spu.Quantity('nanonewton', abbrev='nN')  # noqa: N816
nanonewton.set_global_relative_scale_factor(spu.nano, spu.newton)

micronewton = uN = spu.Quantity('micronewton', abbrev='μN')  # noqa: N816
micronewton.set_global_relative_scale_factor(spu.micro, spu.newton)

millinewton = mN = spu.Quantity('micronewton', abbrev='mN')  # noqa: N816
micronewton.set_global_relative_scale_factor(spu.milli, spu.newton)

# Frequency
kilohertz = KHz = spu.Quantity('kilohertz', abbrev='KHz')
kilohertz.set_global_relative_scale_factor(spu.kilo, spu.hertz)

megahertz = MHz = spu.Quantity('megahertz', abbrev='MHz')
kilohertz.set_global_relative_scale_factor(spu.kilo, spu.hertz)

gigahertz = GHz = spu.Quantity('gigahertz', abbrev='GHz')
gigahertz.set_global_relative_scale_factor(spu.giga, spu.hertz)

terahertz = THz = spu.Quantity('terahertz', abbrev='THz')
terahertz.set_global_relative_scale_factor(spu.tera, spu.hertz)

petahertz = PHz = spu.Quantity('petahertz', abbrev='PHz')
petahertz.set_global_relative_scale_factor(spu.peta, spu.hertz)

exahertz = EHz = spu.Quantity('exahertz', abbrev='EHz')
exahertz.set_global_relative_scale_factor(spu.exa, spu.hertz)

# Pressure
millibar = mbar = spu.Quantity('millibar', abbrev='mbar')
millibar.set_global_relative_scale_factor(spu.milli, spu.bar)

hectopascal = hPa = spu.Quantity('hectopascal', abbrev='hPa')  # noqa: N816
hectopascal.set_global_relative_scale_factor(spu.hecto, spu.pascal)

UNIT_BY_SYMBOL: dict[sp.Symbol, spu.Quantity] = {
	unit.name: unit for unit in spu.__dict__.values() if isinstance(unit, spu.Quantity)
} | {unit.name: unit for unit in globals().values() if isinstance(unit, spu.Quantity)}

UNIT_TO_1: dict[spu.Quantity, 1] = {unit: 1 for unit in UNIT_BY_SYMBOL.values()}
