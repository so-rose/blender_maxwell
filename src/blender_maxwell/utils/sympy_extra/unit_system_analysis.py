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

"""Functions for conversion and casting of `sympy` objects that use units, via unit systems."""

import jax
import sympy.physics.units as spu

from . import units as spux
from .parse_cast import sympy_to_python
from .physical_type import PhysicalType
from .sympy_type import SympyType
from .unit_analysis import get_units
from .unit_systems import UnitSystem


####################
# - Conversion
####################
def strip_unit_system(
	sp_obj: SympyType, unit_system: UnitSystem | None = None
) -> SympyType:
	"""Strip units occurring in the given unit system from the expression.

	Unit stripping is a "dumb" operation: "Substitute any `sympy` object in `unit_system.values()` with `1`".
	Obviously, the semantic correctness of this operation depends entirely on _the units adding no semantic meaning to the expression_.

	Notes:
		You should probably use `scale_to_unit_system()` or `convert_to_unit_system()`.
	"""
	if unit_system is None:
		return sp_obj.subs(spux.UNIT_TO_1)

	return sp_obj.subs({unit: 1 for unit in unit_system.values() if unit is not None})


def convert_to_unit_system(
	sp_obj: SympyType, unit_system: UnitSystem | None
) -> SympyType:
	"""Convert an expression to the units of a given unit system."""
	if unit_system is None:
		return sp_obj

	return spu.convert_to(
		sp_obj,
		{unit_system[PhysicalType.from_unit(unit)] for unit in get_units(sp_obj)},
	)


####################
# - Casting
####################
def scale_to_unit_system(
	sp_obj: SympyType,
	unit_system: UnitSystem | None,
	use_jax_array: bool = False,
) -> int | float | complex | tuple | jax.Array:
	"""Convert an expression to the units of a given unit system, then strip all units of the unit system.

	Afterwards, it is converted to an appropriate Python type.

	Notes:
		For stability, and performance, reasons, this should only be used at the very last stage.

		Regarding performance: **This is not a fast function**.

	Parameters:
		sp_obj: An arbitrary sympy object, presumably with units.
		unit_system: A unit system mapping `PhysicalType` to particular choices of (compound) units.
			Note that, in this context, only `unit_system.values()` is used.

	Returns:
		An appropriate pure Python type, after scaling to the unit system and stripping all units away.

		If the returned type is array-like, and `use_jax_array` is specified, then (and **only** then) will a `jax.Array` be returned instead of a nested `tuple`.
	"""
	return sympy_to_python(
		strip_unit_system(convert_to_unit_system(sp_obj, unit_system), unit_system),
		use_jax_array=use_jax_array,
	)
