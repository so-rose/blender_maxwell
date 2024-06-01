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

"""Functions for characterizaiton, conversion and casting of `sympy` objects that use units."""

import functools

import sympy as sp
import sympy.physics.units as spu

from . import units as spux
from .parse_cast import sympy_to_python
from .sympy_type import SympyType


####################
# - Unit Characterization
####################
## TODO: Caching w/srepr'ed expression.
## TODO: An LFU cache could do better than an LRU.
def uses_units(sp_obj: SympyType) -> bool:
	"""Determines if an expression uses any units.

	Parameters:
		expr: The sympy object that may contain units.

	Returns:
		Whether or not units are present in the object.
	"""
	return sp_obj.has(spu.Quantity)


## TODO: Caching w/srepr'ed expression.
## TODO: An LFU cache could do better than an LRU.
def get_units(expr: sp.Expr) -> set[spu.Quantity]:
	"""Finds all units used by the expression, and returns them as a set.

	No information about _the relationship between units_ is exposed.
	For example, compound units like `spu.meter / spu.second` would be mapped to `{spu.meter, spu.second}`.


	Notes:
		The expression graph is traversed depth-first with `sp.postorder_traversal`, to search for `sp.Quantity` elements.

		The performance is comparable to the performance of `sp.postorder_traversal`, since the **entire expression graph will always be traversed**, with the added overhead of one `isinstance` call per expression-graph-node.

	Parameters:
		expr: The sympy expression that may contain units.

	Returns:
		All units (`spu.Quantity`) used within the expression.
	"""
	return {
		subexpr
		for subexpr in sp.postorder_traversal(expr)
		if isinstance(subexpr, spu.Quantity)
	}


####################
# - Dimensional Characterization
####################
def unit_dim_to_unit_dim_deps(
	unit_dims: SympyType,
) -> dict[spu.dimensions.Dimension, int] | None:
	"""Normalize an expression to a mapping of its dimensional dependencies.

	Comparing the dimensional dependencies of two `unit_dims` is a meaningful way of determining whether they are equivalent.

	Notes:
		We adhere to SI unit conventions when determining dimensional dependencies, to ensure that ex. `freq -> 1/time` equivalences are normalized away.
		This allows the output of this method to be compared meaningfully, to determine whether two dimensional expressions are equivalent.

		We choose to catch a `TypeError`, for cases where dimensional analysis is impossible (especially `+` or `-` between differing dimensions).
		This may have a slight performance penalty.

	Returns:
		The dimensional dependencies of the dimensional expression.

		If such a thing makes no sense, ex. if `+` or `-` is present between differing unit dimensions, then return None.
	"""
	dimsys_SI = spu.systems.si.dimsys_SI

	# Retrieve Dimensional Dependencies
	try:
		return dimsys_SI.get_dimensional_dependencies(unit_dims)

	# Catch TypeError
	## -> Happens if `+` or `-` is in `unit`.
	## -> Generally, it doesn't make sense to add/subtract differing unit dims.
	## -> Thus, when trying to figure out the unit dimension, there isn't one.
	except TypeError:
		return None


def unit_to_unit_dim_deps(
	unit: SympyType,
) -> dict[spu.dimensions.Dimension, int] | None:
	"""Deduce the dimensional dependencies of a unit.

	Notes:
		Using `.subs()` to replace `sp.Quantity`s with `spu.dimensions.Dimension`s seems to result in an expression that absolutely refuses to claim that it has anything other than raw `sp.Symbol`s.

		This is extremely problematic - dimensional analysis relies on the arithmetic properties of proper `Dimension` objects.

		For this reason, though we'd rather have a `unit_to_unit_dims()` function, we have not yet found a way to do this.
		Luckily, most of our uses cases seem only to require the dimensional dictionary, which (surprisingly) seems accessible using `unit_dim_to_unit_dim_deps()`.

	"""
	# Retrieve Dimensional Dependencies
	## -> NOTE: .subs() alone seems to produce sp.Symbol atoms.
	## -> This is extremely problematic; `Dims` arithmetic has key properties.
	## -> So we have to go all the way to the dimensional dependencies.
	## -> This isn't really respecting the args, but it seems to work :)
	return unit_dim_to_unit_dim_deps(
		unit.subs({arg: arg.dimension for arg in unit.atoms(spu.Quantity)})
	)


def compare_unit_dims(unit_dim_l: SympyType, unit_dim_r: SympyType) -> bool:
	"""Compare the dimensional dependencies of two unit dimensions.

	Comparing the dimensional dependencies of two `unit_dims` is a meaningful way of determining whether they are equivalent.
	"""
	return unit_dim_to_unit_dim_deps(unit_dim_l) == unit_dim_to_unit_dim_deps(
		unit_dim_r
	)


def compare_units_by_unit_dims(unit_l: SympyType, unit_r: SympyType) -> bool:
	"""Compare two units by their unit dimensions."""
	return unit_to_unit_dim_deps(unit_l) == unit_to_unit_dim_deps(unit_r)


def compare_unit_dim_to_unit_dim_deps(
	unit_dim: SympyType, unit_dim_deps: dict[spu.dimensions.Dimension, int]
) -> bool:
	"""Compare the dimensional dependencies of unit dimensions to pre-defined unit dimensions."""
	return unit_dim_to_unit_dim_deps(unit_dim) == unit_dim_deps


####################
# - Unit Casting
####################
def strip_units(sp_obj: SympyType) -> SympyType:
	"""Strip all units by replacing them to `1`.

	This is a rather unsafe method.
	You probably shouldn't use it.

	Warnings:
		Absolutely no effort is made to determine whether stripping units is a _meaningful thing to do_.

		For example, using `+` expressions of compatible dimension, but different units, is a clear mistake.
		For example, `8*meter + 9*millimeter` strips to `8(1) + 9(1) = 17`, which is a garbage result.

		The **user of this method** must themselves perform appropriate checks on th eobject before stripping units.

	Parameters:
		sp_obj: A sympy object that contains unit symbols.
			**NOTE**: Unit symbols (from `sympy.physics.units`) are not _free_ symbols, in that they are not unknown.
			Nonetheless, they are not _numbers_ either, and thus they cannot be used in a numerical expression.

	Returns:
		The sympy object with all unit symbols replaced by `1`, effectively extracting the unitless part of the object.
	"""
	return sp_obj.subs(spux.UNIT_TO_1)


def convert_to_unit(sp_obj: SympyType, unit: SympyType | None) -> SympyType:
	"""Convert a sympy object to the given unit.

	Supports a unit of `None`, which simply causes the object to have its units stripped.
	"""
	if unit is None:
		return strip_units(sp_obj)
	return spu.convert_to(sp_obj, unit)

	# msg = f'Sympy object "{sp_obj}" was scaled to the unit "{unit}" with the expectation that the result would be unitless, but the result "{unitless_expr}" has units "{get_units(unitless_expr)}"'
	# raise ValueError(msg)


## TODO: Include sympy_to_python in 'scale_to' to match semantics of 'scale_to_unit_system'
## -- Introduce a 'strip_unit
def scale_to_unit(
	sp_obj: SympyType,
	unit: spu.Quantity | None,
	cast_to_pytype: bool = False,
	use_jax_array: bool = False,
) -> SympyType:
	"""Convert an expression that uses units to a different unit, then strip all units, leaving only a unitless `sympy` value.

	This is used whenever the unitless part of an expression is needed, but guaranteed expressed in a particular unit, aka. **unit system normalization**.

	Notes:
		The unitless output is still an `sp.Expr`, which may contain ex. symbols.

		If you know that the output **should** work as a corresponding Python type (ex. `sp.Integer` vs. `int`), but it doesn't, you can use `sympy_to_python()` to produce a pure-Python type.
		In this way, with a little care, broad compatiblity can be bridged between the `sympy.physics.units` unit system and the wider Python ecosystem.

	Parameters:
		expr: The unit-containing expression to convert.
		unit_to: The unit that is converted to.

	Returns:
		The unitless part of `expr`, after scaling the entire expression to `unit`.

	Raises:
		ValueError: If the result of unit-conversion and -stripping still has units, as determined by `uses_units()`.
	"""
	sp_obj_stripped = strip_units(convert_to_unit(sp_obj, unit))
	if cast_to_pytype:
		return sympy_to_python(
			sp_obj_stripped,
			use_jax_array=use_jax_array,
		)
	return sp_obj_stripped


def scaling_factor(
	unit_from: SympyType, unit_to: SympyType
) -> int | float | complex | tuple | None:
	"""Compute the numerical scaling factor imposed on the unitless part of the expression when converting from one unit to another.

	Parameters:
		unit_from: The unit that is converted from.
		unit_to: The unit that is converted to.

	Returns:
		The numerical scaling factor between the two units.

		If the units are incompatible, then we return None.

	Raises:
		ValueError: If the two units don't share a common dimension.
	"""
	if compare_units_by_unit_dims(unit_from, unit_to):
		return scale_to_unit(unit_from, unit_to)
	return None


@functools.cache
def unit_str_to_unit(unit_str: str, optional: bool = False) -> SympyType | None:
	"""Determine the `sympy` unit expression that matches the given unit string.

	Parameters:
		unit_str: A string parseable with `sp.sympify`, which contains a unit expression.
		optional: Whether to return
			**NOTE**: `None` is itself a valid "unit", denoting dimensionlessness, in general.
			Ensure that appropriate checks are performed to account for this nuance.

	Returns:
		The matching `sympy` unit.

	Raises:
		ValueError: When no valid unit can be matched to the unit string, and `optional` is `False`.
	"""
	match unit_str:
		# Special-Case 'degree'
		## -> sp.sympify('degree') produces the sp.degree().
		## -> TODO: Proper Analysis analysis.
		case 'degree':
			unit = spu.degree

		case _:
			unit = sp.sympify(unit_str).subs(spux.UNIT_BY_SYMBOL)

	if uses_units(unit):
		return unit

	if optional:
		return None
	msg = f'No valid unit for unit string {unit_str}'
	raise ValueError(msg)
