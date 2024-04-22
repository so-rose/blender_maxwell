import sympy.physics.units as spu

from blender_maxwell.utils import extra_sympy_units as spux

from .socket_types import SocketType as ST  # noqa: N817

SOCKET_UNITS = {
	ST.PhysicalTime: {
		'default': 'PS',
		'values': {
			'FS': spux.femtosecond,
			'PS': spu.picosecond,
			'NS': spu.nanosecond,
			'MS': spu.microsecond,
			'MLSEC': spu.millisecond,
			'SEC': spu.second,
			'MIN': spu.minute,
			'HOUR': spu.hour,
			'DAY': spu.day,
		},
	},
	ST.PhysicalAngle: {
		'default': 'RADIAN',
		'values': {
			'RADIAN': spu.radian,
			'DEGREE': spu.degree,
			'STERAD': spu.steradian,
			'ANGMIL': spu.angular_mil,
		},
	},
	ST.PhysicalLength: {
		'default': 'UM',
		'values': {
			'PM': spu.picometer,
			'A': spu.angstrom,
			'NM': spu.nanometer,
			'UM': spu.micrometer,
			'MM': spu.millimeter,
			'CM': spu.centimeter,
			'M': spu.meter,
			'INCH': spu.inch,
			'FOOT': spu.foot,
			'YARD': spu.yard,
			'MILE': spu.mile,
		},
	},
	ST.PhysicalArea: {
		'default': 'UM_SQ',
		'values': {
			'PM_SQ': spu.picometer**2,
			'A_SQ': spu.angstrom**2,
			'NM_SQ': spu.nanometer**2,
			'UM_SQ': spu.micrometer**2,
			'MM_SQ': spu.millimeter**2,
			'CM_SQ': spu.centimeter**2,
			'M_SQ': spu.meter**2,
			'INCH_SQ': spu.inch**2,
			'FOOT_SQ': spu.foot**2,
			'YARD_SQ': spu.yard**2,
			'MILE_SQ': spu.mile**2,
		},
	},
	ST.PhysicalVolume: {
		'default': 'UM_CB',
		'values': {
			'PM_CB': spu.picometer**3,
			'A_CB': spu.angstrom**3,
			'NM_CB': spu.nanometer**3,
			'UM_CB': spu.micrometer**3,
			'MM_CB': spu.millimeter**3,
			'CM_CB': spu.centimeter**3,
			'M_CB': spu.meter**3,
			'ML': spu.milliliter,
			'L': spu.liter,
			'INCH_CB': spu.inch**3,
			'FOOT_CB': spu.foot**3,
			'YARD_CB': spu.yard**3,
			'MILE_CB': spu.mile**3,
		},
	},
	ST.PhysicalPoint2D: {
		'default': 'UM',
		'values': {
			'PM': spu.picometer,
			'A': spu.angstrom,
			'NM': spu.nanometer,
			'UM': spu.micrometer,
			'MM': spu.millimeter,
			'CM': spu.centimeter,
			'M': spu.meter,
			'INCH': spu.inch,
			'FOOT': spu.foot,
			'YARD': spu.yard,
			'MILE': spu.mile,
		},
	},
	ST.PhysicalPoint3D: {
		'default': 'UM',
		'values': {
			'PM': spu.picometer,
			'A': spu.angstrom,
			'NM': spu.nanometer,
			'UM': spu.micrometer,
			'MM': spu.millimeter,
			'CM': spu.centimeter,
			'M': spu.meter,
			'INCH': spu.inch,
			'FOOT': spu.foot,
			'YARD': spu.yard,
			'MILE': spu.mile,
		},
	},
	ST.PhysicalSize2D: {
		'default': 'UM',
		'values': {
			'PM': spu.picometer,
			'A': spu.angstrom,
			'NM': spu.nanometer,
			'UM': spu.micrometer,
			'MM': spu.millimeter,
			'CM': spu.centimeter,
			'M': spu.meter,
			'INCH': spu.inch,
			'FOOT': spu.foot,
			'YARD': spu.yard,
			'MILE': spu.mile,
		},
	},
	ST.PhysicalSize3D: {
		'default': 'UM',
		'values': {
			'PM': spu.picometer,
			'A': spu.angstrom,
			'NM': spu.nanometer,
			'UM': spu.micrometer,
			'MM': spu.millimeter,
			'CM': spu.centimeter,
			'M': spu.meter,
			'INCH': spu.inch,
			'FOOT': spu.foot,
			'YARD': spu.yard,
			'MILE': spu.mile,
		},
	},
	ST.PhysicalMass: {
		'default': 'UG',
		'values': {
			'E_REST': spu.electron_rest_mass,
			'DAL': spu.dalton,
			'UG': spu.microgram,
			'MG': spu.milligram,
			'G': spu.gram,
			'KG': spu.kilogram,
			'TON': spu.metric_ton,
		},
	},
	ST.PhysicalSpeed: {
		'default': 'UM_S',
		'values': {
			'PM_S': spu.picometer / spu.second,
			'NM_S': spu.nanometer / spu.second,
			'UM_S': spu.micrometer / spu.second,
			'MM_S': spu.millimeter / spu.second,
			'M_S': spu.meter / spu.second,
			'KM_S': spu.kilometer / spu.second,
			'KM_H': spu.kilometer / spu.hour,
			'FT_S': spu.feet / spu.second,
			'MI_H': spu.mile / spu.hour,
		},
	},
	ST.PhysicalAccelScalar: {
		'default': 'UM_S_SQ',
		'values': {
			'PM_S_SQ': spu.picometer / spu.second**2,
			'NM_S_SQ': spu.nanometer / spu.second**2,
			'UM_S_SQ': spu.micrometer / spu.second**2,
			'MM_S_SQ': spu.millimeter / spu.second**2,
			'M_S_SQ': spu.meter / spu.second**2,
			'KM_S_SQ': spu.kilometer / spu.second**2,
			'FT_S_SQ': spu.feet / spu.second**2,
		},
	},
	ST.PhysicalForceScalar: {
		'default': 'UNEWT',
		'values': {
			'KG_M_S_SQ': spu.kg * spu.m / spu.second**2,
			'NNEWT': spux.nanonewton,
			'UNEWT': spux.micronewton,
			'MNEWT': spux.millinewton,
			'NEWT': spu.newton,
		},
	},
	ST.PhysicalAccel3D: {
		'default': 'UM_S_SQ',
		'values': {
			'PM_S_SQ': spu.picometer / spu.second**2,
			'NM_S_SQ': spu.nanometer / spu.second**2,
			'UM_S_SQ': spu.micrometer / spu.second**2,
			'MM_S_SQ': spu.millimeter / spu.second**2,
			'M_S_SQ': spu.meter / spu.second**2,
			'KM_S_SQ': spu.kilometer / spu.second**2,
			'FT_S_SQ': spu.feet / spu.second**2,
		},
	},
	ST.PhysicalForce3D: {
		'default': 'UNEWT',
		'values': {
			'KG_M_S_SQ': spu.kg * spu.m / spu.second**2,
			'NNEWT': spux.nanonewton,
			'UNEWT': spux.micronewton,
			'MNEWT': spux.millinewton,
			'NEWT': spu.newton,
		},
	},
	ST.PhysicalFreq: {
		'default': 'THZ',
		'values': {
			'HZ': spu.hertz,
			'KHZ': spux.kilohertz,
			'MHZ': spux.megahertz,
			'GHZ': spux.gigahertz,
			'THZ': spux.terahertz,
			'PHZ': spux.petahertz,
			'EHZ': spux.exahertz,
		},
	},
	ST.PhysicalPol: {
		'default': 'RADIAN',
		'values': {
			'RADIAN': spu.radian,
			'DEGREE': spu.degree,
			'STERAD': spu.steradian,
			'ANGMIL': spu.angular_mil,
		},
	},
	ST.MaxwellMedium: {
		'default': 'NM',
		'values': {
			'PM': spu.picometer,  ## c(vac) = wl*freq
			'A': spu.angstrom,
			'NM': spu.nanometer,
			'UM': spu.micrometer,
			'MM': spu.millimeter,
			'CM': spu.centimeter,
			'M': spu.meter,
		},
	},
	ST.MaxwellMonitor: {
		'default': 'NM',
		'values': {
			'PM': spu.picometer,  ## c(vac) = wl*freq
			'A': spu.angstrom,
			'NM': spu.nanometer,
			'UM': spu.micrometer,
			'MM': spu.millimeter,
			'CM': spu.centimeter,
			'M': spu.meter,
		},
	},
}


def unit_to_socket_type(unit: spux.Unit) -> ST:
	"""Returns a SocketType that accepts the given unit.

	Only the unit-compatibility is taken into account; in the case of overlap, several the ordering of `SOCKET_UNITS` determines which is returned.
	This isn't super clean, but it's good enough for our needs right now.

	Returns:
		**The first `SocketType` in `SOCKET_UNITS`, which contains the given unit as a valid possibility.
	"""
	for socket_type, _units in SOCKET_UNITS.items():
		if unit in _units['values'].values():
			return socket_type

	msg = f"Unit {unit} doesn't have an obvious SocketType."
	raise ValueError(msg)
