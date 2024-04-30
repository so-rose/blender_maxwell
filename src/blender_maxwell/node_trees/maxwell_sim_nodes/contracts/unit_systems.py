"""Specifies unit systems for use in the node tree.

Attributes:
	UNITS_BLENDER: A unit system that serves as a reasonable default for the a 3D workspace that interprets the results of electromagnetic simulations.
		**NOTE**: The addon _specifically_ neglects to respect Blender's builtin units.
		In testing, Blender's system was found to be extremely brittle when "going small" like this; in particular, `picosecond`-order time units were impossible to specify functionally.
	UNITS_TIDY3D: A unit system that aligns with Tidy3D's simulator.
		See <https://docs.flexcompute.com/projects/tidy3d/en/latest/faq/docs/faq/What-are-the-units-used-in-the-simulation.html>
"""

import typing as typ

import sympy.physics.units as spu

from blender_maxwell.utils import extra_sympy_units as spux

####################
# - Unit Systems
####################
_PT: typ.TypeAlias = spux.PhysicalType
UNITS_BLENDER: spux.UnitSystem = spux.UNITS_SI | {
	# Global
	_PT.Time: spu.picosecond,
	_PT.Freq: spux.terahertz,
	_PT.AngFreq: spu.radian * spux.terahertz,
	# Cartesian
	_PT.Length: spu.micrometer,
	_PT.Area: spu.micrometer**2,
	_PT.Volume: spu.micrometer**3,
	# Energy
	_PT.PowerFlux: spu.watt / spu.um**2,
	# Electrodynamics
	_PT.CurrentDensity: spu.ampere / spu.um**2,
	_PT.Conductivity: spu.siemens / spu.um,
	_PT.PoyntingVector: spu.watt / spu.um**2,
	_PT.EField: spu.volt / spu.um,
	_PT.HField: spu.ampere / spu.um,
	# Mechanical
	_PT.Vel: spu.um / spu.second,
	_PT.Accel: spu.um / spu.second,
	_PT.Mass: spu.microgram,
	_PT.Force: spux.micronewton,
	# Luminal
	# Optics
	_PT.PoyntingVector: spu.watt / spu.um**2,
}  ## TODO: Load (dynamically?) from addon preferences

UNITS_TIDY3D: spux.UnitSystem = spux.UNITS_SI | {
	# Global
	# Cartesian
	_PT.Length: spu.um,
	_PT.Area: spu.um**2,
	_PT.Volume: spu.um**3,
	# Mechanical
	_PT.Vel: spu.um / spu.second,
	_PT.Accel: spu.um / spu.second,
	# Energy
	_PT.PowerFlux: spu.watt / spu.um**2,
	# Electrodynamics
	_PT.CurrentDensity: spu.ampere / spu.um**2,
	_PT.Conductivity: spu.siemens / spu.um,
	_PT.PoyntingVector: spu.watt / spu.um**2,
	_PT.EField: spu.volt / spu.um,
	_PT.HField: spu.ampere / spu.um,
	# Luminal
	# Optics
	_PT.PoyntingVector: spu.watt / spu.um**2,
	## NOTE: w/o source normalization, EField/HField/Modal amps are * 1/Hz
}
