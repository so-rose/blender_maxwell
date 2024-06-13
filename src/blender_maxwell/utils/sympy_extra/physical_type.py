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

"""Implements `PhysicalType`, a convenient, UI-friendly way of deterministically handling the unit-dimensionality of arbitrary objects."""

import enum
import functools
import typing as typ

import sympy.physics.units as spu

from blender_maxwell import contracts as ct

from ..staticproperty import staticproperty
from . import units as spux
from .math_type import MathType
from .sympy_expr import Unit
from .sympy_type import SympyType
from .unit_analysis import (
	compare_unit_dim_to_unit_dim_deps,
	compare_unit_dims,
	unit_to_unit_dim_deps,
)


####################
# - Unit Dimensions
####################
class DimsMeta(type):
	"""Metaclass allowing an implementing (ideally empty) class to access `spu.definitions.dimension_definitions` attributes directly via its own attribute."""

	def __getattr__(cls, attr: str) -> spu.Dimension:
		"""Alias for `spu.definitions.dimension_definitions.*` (isn't that a mouthful?).

		Raises:
			AttributeError: If the name cannot be found.
		"""
		if (
			attr in spu.definitions.dimension_definitions.__dir__()
			and not attr.startswith('__')
		):
			return getattr(spu.definitions.dimension_definitions, attr)

		raise AttributeError(name=attr, obj=Dims)


class Dims(metaclass=DimsMeta):
	"""Access `sympy.physics.units` dimensions with less hassle.

	Any unit dimension available in `sympy.physics.units.definitions.dimension_definitions` can be accessed as an attribute of `Dims`.

	An `AttributeError` is raised if the unit cannot be found in `sympy`.

	Examples:
		The objects returned are a direct alias to `sympy`, with less hassle:
		```python
		assert Dims.length == (
			sympy.physics.units.definitions.dimension_definitions.length
		)
		```
	"""


####################
# - Physical Type
####################
class PhysicalType(enum.StrEnum):
	"""An identifier of unit dimensionality with many useful properties."""

	# Unitless
	NonPhysical = enum.auto()

	# Global
	Time = enum.auto()
	Angle = enum.auto()
	SolidAngle = enum.auto()
	## TODO: Some kind of 3D-specific orientation ex. a quaternion
	Freq = enum.auto()
	AngFreq = enum.auto()  ## rad*hertz
	# Cartesian
	Length = enum.auto()
	Area = enum.auto()
	Volume = enum.auto()
	# Mechanical
	Vel = enum.auto()
	Accel = enum.auto()
	Mass = enum.auto()
	Force = enum.auto()
	Pressure = enum.auto()
	# Energy
	Work = enum.auto()  ## joule
	Power = enum.auto()  ## watt
	PowerFlux = enum.auto()  ## watt
	Temp = enum.auto()
	# Electrodynamics
	Current = enum.auto()  ## ampere
	CurrentDensity = enum.auto()
	Charge = enum.auto()  ## coulomb
	Voltage = enum.auto()
	Capacitance = enum.auto()  ## farad
	Impedance = enum.auto()  ## ohm
	Conductance = enum.auto()  ## siemens
	Conductivity = enum.auto()  ## siemens / length
	MFlux = enum.auto()  ## weber
	MFluxDensity = enum.auto()  ## tesla
	Inductance = enum.auto()  ## henry
	EField = enum.auto()
	HField = enum.auto()
	# Luminal
	LumIntensity = enum.auto()
	LumFlux = enum.auto()
	Illuminance = enum.auto()

	####################
	# - Unit Dimensions
	####################
	@functools.cached_property
	def unit_dim(self) -> SympyType:
		"""The unit dimension expression associated with the `PhysicalType`.

		A `PhysicalType` is, in its essence, merely an identifier for a particular unit dimension expression.
		"""
		PT = PhysicalType
		return {
			PT.NonPhysical: None,
			# Global
			PT.Time: Dims.time,
			PT.Angle: Dims.angle,
			PT.SolidAngle: spu.steradian.dimension,  ## MISSING
			PT.Freq: Dims.frequency,
			PT.AngFreq: Dims.angle * Dims.frequency,
			# Cartesian
			PT.Length: Dims.length,
			PT.Area: Dims.length**2,
			PT.Volume: Dims.length**3,
			# Mechanical
			PT.Vel: Dims.length / Dims.time,
			PT.Accel: Dims.length / Dims.time**2,
			PT.Mass: Dims.mass,
			PT.Force: Dims.force,
			PT.Pressure: Dims.pressure,
			# Energy
			PT.Work: Dims.energy,
			PT.Power: Dims.power,
			PT.PowerFlux: Dims.power / Dims.length**2,
			PT.Temp: Dims.temperature,
			# Electrodynamics
			PT.Current: Dims.current,
			PT.CurrentDensity: Dims.current / Dims.length**2,
			PT.Charge: Dims.charge,
			PT.Voltage: Dims.voltage,
			PT.Capacitance: Dims.capacitance,
			PT.Impedance: Dims.impedance,
			PT.Conductance: Dims.conductance,
			PT.Conductivity: Dims.conductance / Dims.length,
			PT.MFlux: Dims.magnetic_flux,
			PT.MFluxDensity: Dims.magnetic_density,
			PT.Inductance: Dims.inductance,
			PT.EField: Dims.voltage / Dims.length,
			PT.HField: Dims.current / Dims.length,
			# Luminal
			PT.LumIntensity: Dims.luminous_intensity,
			PT.LumFlux: Dims.luminous_intensity * spu.steradian.dimension,
			PT.Illuminance: Dims.luminous_intensity / Dims.length**2,
		}[self]

	@staticproperty
	def unit_dims() -> dict[typ.Self, SympyType]:
		"""All unit dimensions supported by all `PhysicalType`s."""
		return {
			physical_type: physical_type.unit_dim
			for physical_type in list(PhysicalType)
		}

	####################
	# - Convenience Properties
	####################
	@functools.cached_property
	def default_unit(self) -> list[Unit]:
		"""Subjective choice of 'default' unit from `self.valid_units`.

		There is no requirement to use this.
		"""
		PT = PhysicalType
		return {
			PT.NonPhysical: None,
			# Global
			PT.Time: spu.picosecond,
			PT.Angle: spu.radian,
			PT.SolidAngle: spu.steradian,
			PT.Freq: spux.terahertz,
			PT.AngFreq: spu.radian * spux.terahertz,
			# Cartesian
			PT.Length: spu.micrometer,
			PT.Area: spu.um**2,
			PT.Volume: spu.um**3,
			# Mechanical
			PT.Vel: spu.um / spu.second,
			PT.Accel: spu.um / spu.second,
			PT.Mass: spu.microgram,
			PT.Force: spux.micronewton,
			PT.Pressure: spux.millibar,
			# Energy
			PT.Work: spu.joule,
			PT.Power: spu.watt,
			PT.PowerFlux: spu.watt / spu.meter**2,
			PT.Temp: spu.kelvin,
			# Electrodynamics
			PT.Current: spu.ampere,
			PT.CurrentDensity: spu.ampere / spu.meter**2,
			PT.Charge: spu.coulomb,
			PT.Voltage: spu.volt,
			PT.Capacitance: spu.farad,
			PT.Impedance: spu.ohm,
			PT.Conductance: spu.siemens,
			PT.Conductivity: spu.siemens / spu.micrometer,
			PT.MFlux: spu.weber,
			PT.MFluxDensity: spu.tesla,
			PT.Inductance: spu.henry,
			PT.EField: spu.volt / spu.micrometer,
			PT.HField: spu.ampere / spu.micrometer,
			# Luminal
			PT.LumIntensity: spu.candela,
			PT.LumFlux: spu.candela * spu.steradian,
			PT.Illuminance: spu.candela / spu.meter**2,
		}[self]

	####################
	# - Creation
	####################
	@staticmethod
	def from_unit(
		unit: Unit | None, optional: bool = False, optional_nonphy: bool = False
	) -> typ.Self | None:
		"""Attempt to determine a matching `PhysicalType` from a unit.

		NOTE: It is not guaranteed that `unit` is within `valid_units`, only that it can be converted to any unit in `valid_units`.

		Returns:
			The matched `PhysicalType`.

			If none could be matched, then either return `None` (if `optional` is set) or error.

		Raises:
			ValueError: If no `PhysicalType` could be matched, and `optional` is `False`.
		"""
		if unit is None:
			return PhysicalType.NonPhysical

		## TODO: This enough?
		if unit in [spu.radian, spu.degree]:
			return PhysicalType.Angle

		unit_dim_deps = unit_to_unit_dim_deps(unit)
		if unit_dim_deps is not None:
			for physical_type, candidate_unit_dim in PhysicalType.unit_dims.items():
				if compare_unit_dim_to_unit_dim_deps(candidate_unit_dim, unit_dim_deps):
					return physical_type

		if optional:
			if optional_nonphy:
				return PhysicalType.NonPhysical
			return None
		msg = f'Could not determine PhysicalType for {unit}'
		raise ValueError(msg)

	@staticmethod
	def from_unit_dim(
		unit_dim: SympyType | None, optional: bool = False
	) -> typ.Self | None:
		"""Attempts to match an arbitrary unit dimension expression to a corresponding `PhysicalType`.

		For comparing arbitrary unit dimensions (via expressions of `spu.dimensions.Dimension`), it is critical that equivalent dimensions are also compared as equal (ex. `mass*length/time^2 == force`).
		To do so, we employ the `SI` unit conventions, for extracting the fundamental dimensional dependencies of unit dimension expressions.

		Returns:
			The matched `PhysicalType`.

			If none could be matched, then either return `None` (if `optional` is set) or error.

		Raises:
			ValueError: If no `PhysicalType` could be matched, and `optional` is `False`.
		"""
		for physical_type, candidate_unit_dim in PhysicalType.unit_dims.items():
			if compare_unit_dims(unit_dim, candidate_unit_dim):
				return physical_type

		if optional:
			return None
		msg = f'Could not determine PhysicalType for {unit_dim}'
		raise ValueError(msg)

	####################
	# - Valid Properties
	####################
	@functools.cached_property
	def valid_units(self) -> list[Unit | None]:
		"""Retrieve an ordered (by subjective usefulness) list of units for this physical type.

		`None` denotes "no units are valid".

		Warnings:
			**Altering the order of units hard-breaks backwards compatibility**, since enums based on it only keep an integer index.

		Notes:
			The order in which valid units are declared is the exact same order that UI dropdowns display them.
		"""
		PT = PhysicalType
		return {
			PT.NonPhysical: [None],
			# Global
			PT.Time: [
				spu.picosecond,
				spux.femtosecond,
				spu.nanosecond,
				spu.microsecond,
				spu.millisecond,
				spu.second,
				spu.minute,
				spu.hour,
				spu.day,
			],
			PT.Angle: [
				spu.radian,
				spu.degree,
			],
			PT.SolidAngle: [
				spu.steradian,
			],
			PT.Freq: (
				_valid_freqs := [
					spux.terahertz,
					spu.hertz,
					spux.kilohertz,
					spux.megahertz,
					spux.gigahertz,
					spux.petahertz,
					spux.exahertz,
				]
			),
			PT.AngFreq: [spu.radian * _unit for _unit in _valid_freqs],
			# Cartesian
			PT.Length: (
				_valid_lens := [
					spu.micrometer,
					spu.nanometer,
					spu.picometer,
					spu.angstrom,
					spu.millimeter,
					spu.centimeter,
					spu.meter,
					spu.inch,
					spu.foot,
					spu.yard,
					spu.mile,
				]
			),
			PT.Area: [_unit**2 for _unit in _valid_lens],
			PT.Volume: [_unit**3 for _unit in _valid_lens],
			# Mechanical
			PT.Vel: [_unit / spu.second for _unit in _valid_lens],
			PT.Accel: [_unit / spu.second**2 for _unit in _valid_lens],
			PT.Mass: [
				spu.kilogram,
				spu.electron_rest_mass,
				spu.dalton,
				spu.microgram,
				spu.milligram,
				spu.gram,
				spu.metric_ton,
			],
			PT.Force: [
				spux.micronewton,
				spux.nanonewton,
				spux.millinewton,
				spu.newton,
				spu.kg * spu.meter / spu.second**2,
			],
			PT.Pressure: [
				spu.bar,
				spux.millibar,
				spu.pascal,
				spux.hectopascal,
				spu.atmosphere,
				spu.psi,
				spu.mmHg,
				spu.torr,
			],
			# Energy
			PT.Work: [
				spu.joule,
				spu.electronvolt,
			],
			PT.Power: [
				spu.watt,
			],
			PT.PowerFlux: [
				spu.watt / spu.meter**2,
			],
			PT.Temp: [
				spu.kelvin,
			],
			# Electrodynamics
			PT.Current: [
				spu.ampere,
			],
			PT.CurrentDensity: [
				spu.ampere / spu.meter**2,
			],
			PT.Charge: [
				spu.coulomb,
			],
			PT.Voltage: [
				spu.volt,
			],
			PT.Capacitance: [
				spu.farad,
			],
			PT.Impedance: [
				spu.ohm,
			],
			PT.Conductance: [
				spu.siemens,
			],
			PT.Conductivity: [
				spu.siemens / spu.micrometer,
				spu.siemens / spu.meter,
			],
			PT.MFlux: [
				spu.weber,
			],
			PT.MFluxDensity: [
				spu.tesla,
			],
			PT.Inductance: [
				spu.henry,
			],
			PT.EField: [
				spu.volt / spu.micrometer,
				spu.volt / spu.meter,
			],
			PT.HField: [
				spu.ampere / spu.micrometer,
				spu.ampere / spu.meter,
			],
			# Luminal
			PT.LumIntensity: [
				spu.candela,
			],
			PT.LumFlux: [
				spu.candela * spu.steradian,
			],
			PT.Illuminance: [
				spu.candela / spu.meter**2,
			],
		}[self]

	@functools.cached_property
	def valid_shapes(self) -> list[typ.Literal[(3,), (2,), ()] | None]:
		"""All shapes with physical meaning in the context of a particular unit dimension.

		Don't use with `NonPhysical`.
		"""
		PT = PhysicalType
		overrides = {
			# Cartesian
			PT.Length: [(), (2,), (3,)],
			# Mechanical
			PT.Vel: [(), (2,), (3,)],
			PT.Accel: [(), (2,), (3,)],
			PT.Force: [(), (2,), (3,)],
			# Energy
			PT.Work: [(), (2,), (3,)],
			PT.PowerFlux: [(), (2,), (3,)],
			# Electrodynamics
			PT.CurrentDensity: [(), (2,), (3,)],
			PT.MFluxDensity: [(), (2,), (3,)],
			PT.EField: [(), (2,), (3,)],
			PT.HField: [(), (2,), (3,)],
			# Luminal
			PT.LumFlux: [(), (2,), (3,)],
		}

		return overrides.get(self, [()])

	@functools.cached_property
	def valid_mathtypes(self) -> list[MathType]:
		"""Returns a list of valid mathematical types, especially whether it can be real- or complex-valued.

		Generally, all unit quantities are real, in the algebraic mathematical sense.
		However, in electrodynamics especially, it becomes enormously useful to bake in a _rotational component_ as an imaginary value, be it simply to model phase or oscillation-oriented dampening.
		This imaginary part has physical meaning, which can be expressed using the same mathematical formalism associated with unit systems.
		In general, the value is a phasor.

		While it is difficult to arrive at a well-defined way of saying, "this is when a quantity is complex", an attempt has been made to form a sensible baseline based on when phasor math may apply.

		Notes:
			- **Freq**/**AngFreq**: The imaginary part represents growth/dampening of the oscillation.
			- **Current**/**Voltage**: The imaginary part represents the phase.
				This also holds for any downstream units.
			- **Charge**: Generally, it is real.
				However, an imaginary phase term seems to have research applications when dealing with high-order harmonics in high-energy pulsed lasers: <https://iopscience.iop.org/article/10.1088/1361-6455/aac787>
			- **Conductance**: The imaginary part represents the extinction, in the Drude-model sense.

		"""
		MT = MathType
		PT = PhysicalType
		overrides = {
			PT.NonPhysical: list(MT),  ## Support All
			# Cartesian
			PT.Freq: [MT.Real, MT.Complex],  ## Im -> Growth/Damping
			PT.AngFreq: [MT.Real, MT.Complex],  ## Im -> Growth/Damping
			# Mechanical
			# Energy
			# Electrodynamics
			PT.Current: [MT.Real, MT.Complex],  ## Im -> Phase
			PT.CurrentDensity: [MT.Real, MT.Complex],  ## Im -> Phase
			PT.Charge: [MT.Real, MT.Complex],  ## Im -> Phase
			PT.Voltage: [MT.Real, MT.Complex],  ## Im -> Phase
			PT.Capacitance: [MT.Real, MT.Complex],  ## Im -> Phase
			PT.Impedance: [MT.Real, MT.Complex],  ## Im -> Reactance
			PT.Inductance: [MT.Real, MT.Complex],  ## Im -> Extinction
			PT.Conductance: [MT.Real, MT.Complex],  ## Im -> Extinction
			PT.Conductivity: [MT.Real, MT.Complex],  ## Im -> Extinction
			PT.MFlux: [MT.Real, MT.Complex],  ## Im -> Phase
			PT.MFluxDensity: [MT.Real, MT.Complex],  ## Im -> Phase
			PT.EField: [MT.Real, MT.Complex],  ## Im -> Phase
			PT.HField: [MT.Real, MT.Complex],  ## Im -> Phase
			# Luminal
		}

		return overrides.get(self, [MT.Real])

	####################
	# - UI
	####################
	@staticmethod
	def to_name(value: typ.Self) -> str:
		"""A human-readable UI-oriented name for a physical type."""
		if value is PhysicalType.NonPhysical:
			return 'Unitless'
		return PhysicalType(value).name

	@staticmethod
	def to_icon(_: typ.Self) -> str:
		"""No icons."""
		return ''

	def bl_enum_element(self, i: int) -> ct.BLEnumElement:
		"""Given an integer index, generate an element that conforms to the requirements of `bpy.props.EnumProperty.items`."""
		PT = PhysicalType
		return (
			str(self),
			PT.to_name(self),
			PT.to_name(self),
			PT.to_icon(self),
			i,
		)

	@functools.cached_property
	def color(self):
		"""A color corresponding to the physical type.

		The color selections were initially generated using AI, as this is a rote task that's better adjusted than invented.
		The LLM provided the following rationale for its choices:

		> Non-Physical: Grey signifies neutrality and non-physical nature.
		> Global:
		>    Time: Blue is often associated with calmness and the passage of time.
		>    Angle and Solid Angle: Different shades of blue and cyan suggest angular dimensions and spatial aspects.
		>    Frequency and Angular Frequency: Darker shades of blue to maintain the link to time.
		> Cartesian:
		>    Length, Area, Volume: Shades of green to represent spatial dimensions, with intensity increasing with dimension.
		> Mechanical:
		>    Velocity and Acceleration: Red signifies motion and dynamics, with lighter reds for related quantities.
		>    Mass: Dark red for the fundamental property.
		>    Force and Pressure: Shades of red indicating intensity.
		> Energy:
		>    Work and Power: Orange signifies energy transformation, with lighter oranges for related quantities.
		>    Temperature: Yellow for heat.
		> Electrodynamics:
		>    Current and related quantities: Cyan shades indicating flow.
		>    Voltage, Capacitance: Greenish and blueish cyan for electrical potential.
		>    Impedance, Conductance, Conductivity: Purples and magentas to signify resistance and conductance.
		>    Magnetic properties: Magenta shades for magnetism.
		>    Electric Field: Light blue.
		>    Magnetic Field: Grey, as it can be considered neutral in terms of direction.
		> Luminal:
		>    Luminous properties: Yellows to signify light and illumination.
		>
		> This color mapping helps maintain intuitive connections for users interacting with these physical types.
		"""
		PT = PhysicalType
		return {
			PT.NonPhysical: (0.75, 0.75, 0.75, 1.0),  # Light Grey: Non-physical
			# Global
			PT.Time: (0.5, 0.5, 1.0, 1.0),  # Light Blue: Time
			PT.Angle: (0.5, 0.75, 1.0, 1.0),  # Light Blue: Angle
			PT.SolidAngle: (0.5, 0.75, 0.75, 1.0),  # Light Cyan: Solid Angle
			PT.Freq: (0.5, 0.5, 0.9, 1.0),  # Light Blue: Frequency
			PT.AngFreq: (0.5, 0.5, 0.8, 1.0),  # Light Blue: Angular Frequency
			# Cartesian
			PT.Length: (0.5, 1.0, 0.5, 1.0),  # Light Green: Length
			PT.Area: (0.6, 1.0, 0.6, 1.0),  # Light Green: Area
			PT.Volume: (0.7, 1.0, 0.7, 1.0),  # Light Green: Volume
			# Mechanical
			PT.Vel: (1.0, 0.5, 0.5, 1.0),  # Light Red: Velocity
			PT.Accel: (1.0, 0.6, 0.6, 1.0),  # Light Red: Acceleration
			PT.Mass: (0.75, 0.5, 0.5, 1.0),  # Light Red: Mass
			PT.Force: (0.9, 0.5, 0.5, 1.0),  # Light Red: Force
			PT.Pressure: (1.0, 0.7, 0.7, 1.0),  # Light Red: Pressure
			# Energy
			PT.Work: (1.0, 0.75, 0.5, 1.0),  # Light Orange: Work
			PT.Power: (1.0, 0.85, 0.5, 1.0),  # Light Orange: Power
			PT.PowerFlux: (1.0, 0.8, 0.6, 1.0),  # Light Orange: Power Flux
			PT.Temp: (1.0, 1.0, 0.5, 1.0),  # Light Yellow: Temperature
			# Electrodynamics
			PT.Current: (0.5, 1.0, 1.0, 1.0),  # Light Cyan: Current
			PT.CurrentDensity: (0.5, 0.9, 0.9, 1.0),  # Light Cyan: Current Density
			PT.Charge: (0.5, 0.85, 0.85, 1.0),  # Light Cyan: Charge
			PT.Voltage: (0.5, 1.0, 0.75, 1.0),  # Light Greenish Cyan: Voltage
			PT.Capacitance: (0.5, 0.75, 1.0, 1.0),  # Light Blueish Cyan: Capacitance
			PT.Impedance: (0.6, 0.5, 0.75, 1.0),  # Light Purple: Impedance
			PT.Conductance: (0.7, 0.5, 0.8, 1.0),  # Light Purple: Conductance
			PT.Conductivity: (0.8, 0.5, 0.9, 1.0),  # Light Purple: Conductivity
			PT.MFlux: (0.75, 0.5, 0.75, 1.0),  # Light Magenta: Magnetic Flux
			PT.MFluxDensity: (
				0.85,
				0.5,
				0.85,
				1.0,
			),  # Light Magenta: Magnetic Flux Density
			PT.Inductance: (0.8, 0.5, 0.8, 1.0),  # Light Magenta: Inductance
			PT.EField: (0.75, 0.75, 1.0, 1.0),  # Light Blue: Electric Field
			PT.HField: (0.75, 0.75, 0.75, 1.0),  # Light Grey: Magnetic Field
			# Luminal
			PT.LumIntensity: (1.0, 0.95, 0.5, 1.0),  # Light Yellow: Luminous Intensity
			PT.LumFlux: (1.0, 0.95, 0.6, 1.0),  # Light Yellow: Luminous Flux
			PT.Illuminance: (1.0, 1.0, 0.75, 1.0),  # Pale Yellow: Illuminance
		}[self]
