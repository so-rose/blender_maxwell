import typing as typ

import sympy.physics.units as spu

from ....utils import extra_sympy_units as spux
from ....utils.pydantic_sympy import SympyExpr
from .socket_types import SocketType as ST  # noqa: N817
from .socket_units import SOCKET_UNITS


def _socket_units(socket_type):
	return SOCKET_UNITS[socket_type]['values']


UnitSystem: typ.TypeAlias = dict[ST, SympyExpr]
####################
# - Unit Systems
####################
UNITS_BLENDER: UnitSystem = {
	socket_type: _socket_units(socket_type)[socket_unit_prop]
	for socket_type, socket_unit_prop in {
		ST.PhysicalTime: spu.picosecond,
		ST.PhysicalAngle: spu.radian,
		ST.PhysicalLength: spu.micrometer,
		ST.PhysicalArea: spu.micrometer**2,
		ST.PhysicalVolume: spu.micrometer**3,
		ST.PhysicalPoint2D: spu.micrometer,
		ST.PhysicalPoint3D: spu.micrometer,
		ST.PhysicalSize2D: spu.micrometer,
		ST.PhysicalSize3D: spu.micrometer,
		ST.PhysicalMass: spu.microgram,
		ST.PhysicalSpeed: spu.um / spu.second,
		ST.PhysicalAccelScalar: spu.um / spu.second**2,
		ST.PhysicalForceScalar: spu.micronewton,
		ST.PhysicalAccel3D: spu.um / spu.second**2,
		ST.PhysicalForce3D: spu.micronewton,
		ST.PhysicalFreq: spu.terahertz,
		ST.PhysicalPol: spu.radian,
	}.items()
}  ## TODO: Load (dynamically?) from addon preferences

UNITS_TIDY3D: UnitSystem = {
	socket_type: _socket_units(socket_type)[socket_unit_prop]
	for socket_type, socket_unit_prop in {
		ST.PhysicalTime: spu.picosecond,
		ST.PhysicalAngle: spu.radian,
		ST.PhysicalLength: spu.micrometer,
		ST.PhysicalArea: spu.micrometer**2,
		ST.PhysicalVolume: spu.micrometer**3,
		ST.PhysicalPoint2D: spu.micrometer,
		ST.PhysicalPoint3D: spu.micrometer,
		ST.PhysicalSize2D: spu.micrometer,
		ST.PhysicalSize3D: spu.micrometer,
		ST.PhysicalMass: spu.microgram,
		ST.PhysicalSpeed: spu.um / spu.second,
		ST.PhysicalAccelScalar: spu.um / spu.second**2,
		ST.PhysicalForceScalar: spu.micronewton,
		ST.PhysicalAccel3D: spu.um / spu.second**2,
		ST.PhysicalForce3D: spu.micronewton,
		ST.PhysicalFreq: spu.terahertz,
		ST.PhysicalPol: spu.radian,
	}.items()
}