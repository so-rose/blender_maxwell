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

"""Defines a common unit system representation, as well as a few of the most common / useful unit systems.

Attributes:
	UnitSystem: Type of a unit system representation, as an exhaustive mapping from `PhysicalType` to a unit expression.
		**Compatibility between `PhysicalType` and unit must be manually guaranteed** when defining new unit systems.
	UNITS_SI: Pre-defined go-to choice of unit system, which can also be a useful base to build other unit systems on.
"""

import typing as typ

import sympy.physics.units as spu

from . import units as spux
from .physical_type import PhysicalType as PT  # noqa: N817
from .sympy_expr import Unit

####################
# - Unit System Representation
####################
UnitSystem: typ.TypeAlias = dict[PT, Unit]

####################
# - Standard Unit Systems
####################
UNITS_SI: UnitSystem = {
	PT.NonPhysical: None,
	# Global
	PT.Time: spu.second,
	PT.Angle: spu.radian,
	PT.SolidAngle: spu.steradian,
	PT.Freq: spu.hertz,
	PT.AngFreq: spu.radian * spu.hertz,
	# Cartesian
	PT.Length: spu.meter,
	PT.Area: spu.meter**2,
	PT.Volume: spu.meter**3,
	# Mechanical
	PT.Vel: spu.meter / spu.second,
	PT.Accel: spu.meter / spu.second**2,
	PT.Mass: spu.kilogram,
	PT.Force: spu.newton,
	# Energy
	PT.Work: spu.joule,
	PT.Power: spu.watt,
	PT.PowerFlux: spu.watt / spu.meter**2,
	PT.Temp: spu.kelvin,
	# Electrodynamics
	PT.Current: spu.ampere,
	PT.CurrentDensity: spu.ampere / spu.meter**2,
	PT.Voltage: spu.volt,
	PT.Capacitance: spu.farad,
	PT.Impedance: spu.ohm,
	PT.Conductance: spu.siemens,
	PT.Conductivity: spu.siemens / spu.meter,
	PT.MFlux: spu.weber,
	PT.MFluxDensity: spu.tesla,
	PT.Inductance: spu.henry,
	PT.EField: spu.volt / spu.meter,
	PT.HField: spu.ampere / spu.meter,
	# Luminal
	PT.LumIntensity: spu.candela,
	PT.LumFlux: spux.lumen,
	PT.Illuminance: spu.lux,
}
