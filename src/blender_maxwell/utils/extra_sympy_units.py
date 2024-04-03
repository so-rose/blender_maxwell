import functools
import itertools
import typing as typ

from . import pydeps

with pydeps.syspath_from_bpy_prefs():
	import sympy as sp
	import sympy.physics.units as spu


####################
# - Useful Methods
####################
def uses_units(expression: sp.Expr) -> bool:
	## TODO: An LFU cache could do better than an LRU.
	"""Checks if an expression uses any units (`Quantity`)."""
	for arg in sp.preorder_traversal(expression):
		if isinstance(arg, spu.Quantity):
			return True
	return False


# Function to return a set containing all units used in the expression
def get_units(expression: sp.Expr):
	## TODO: An LFU cache could do better than an LRU.
	"""Gets all the units of an expression (as `Quantity`)."""
	return {
		arg
		for arg in sp.preorder_traversal(expression)
		if isinstance(arg, spu.Quantity)
	}


####################
# - Time
####################
femtosecond = fs = spu.Quantity('femtosecond', abbrev='fs')
femtosecond.set_global_relative_scale_factor(spu.femto, spu.second)


####################
# - Force
####################
# Newton
nanonewton = nN = spu.Quantity('nanonewton', abbrev='nN')
nanonewton.set_global_relative_scale_factor(spu.nano, spu.newton)

micronewton = uN = spu.Quantity('micronewton', abbrev='μN')
micronewton.set_global_relative_scale_factor(spu.micro, spu.newton)

millinewton = mN = spu.Quantity('micronewton', abbrev='mN')
micronewton.set_global_relative_scale_factor(spu.milli, spu.newton)

####################
# - Frequency
####################
# Hertz
kilohertz = kHz = spu.Quantity('kilohertz', abbrev='kHz')
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

####################
# - Sympy Expression Typing
####################
ALL_UNIT_SYMBOLS = {
	unit.abbrev: unit
	for unit in spu.__dict__.values()
	if isinstance(unit, spu.Quantity)
} | {unit.abbrev: unit for unit in globals().values() if isinstance(unit, spu.Quantity)}


@functools.lru_cache(maxsize=4096)
def parse_abbrev_symbols_to_units(expr: sp.Basic) -> sp.Basic:
	return expr.subs(ALL_UNIT_SYMBOLS)


####################
# - Units <-> Scalars
####################
def scale_to_unit(expr: sp.Expr, unit: spu.Quantity) -> typ.Any:
	## TODO: An LFU cache could do better than an LRU.
	unitless_expr = spu.convert_to(expr, unit) / unit
	if not uses_units(unitless_expr):
		return unitless_expr

	msg = f'Expression "{expr}" was scaled to the unit "{unit}" with the expectation that the result would be unitless, but the result "{unitless_expr}" has units "{get_units(unitless_expr)}"'
	raise ValueError(msg)


####################
# - Sympy <-> Scalars
####################
def sympy_to_python(scalar: sp.Basic) -> int | float | complex | tuple | list:
	"""Convert a scalar sympy expression to the directly corresponding Python type.

	Arguments:
		scalar: A sympy expression that has no symbols, but is expressed as a Sympy type.
			For expressions that are equivalent to a scalar (ex. "(2a + a)/a"), you must simplify the expression with ex. `sp.simplify()` before passing to this parameter.

	Returns:
		A pure Python type that directly corresponds to the input scalar expression.
	"""
	## TODO: If there are symbols, we could simplify.
	## - Someone has to do it somewhere, might as well be here.
	## - ...Since we have all the information we need.
	if isinstance(scalar, sp.MatrixBase):
		list_2d = [[sympy_to_python(el) for el in row] for row in scalar.tolist()]

		# Detect Row / Column Vector
		## When it's "actually" a 1D structure, flatten and return as tuple.
		if 1 in scalar.shape:
			return tuple(itertools.chain.from_iterable(list_2d))

		return list_2d
	if scalar.is_integer:
		return int(scalar)
	if scalar.is_rational or scalar.is_real:
		return float(scalar)
	if scalar.is_complex:
		return complex(scalar)

	msg = f'Cannot convert sympy scalar expression "{scalar}" to a Python type. Check the assumptions on the expr (current expr assumptions: "{scalar._assumptions}")'  # noqa: SLF001
	raise ValueError(msg)
