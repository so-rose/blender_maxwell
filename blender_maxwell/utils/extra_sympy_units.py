import sympy as sp
import sympy.physics.units as spu

####################
# - Useful Methods
####################
def uses_units(expression: sp.Expr) -> bool:
	"""Checks if an expression uses any units (`Quantity`)."""
	
	for arg in sp.preorder_traversal(expression):
		if isinstance(arg, spu.Quantity):
			return True
	return False

# Function to return a set containing all units used in the expression
def get_units(expression: sp.Expr):
	"""Gets all the units of an expression (as `Quantity`)."""
	
	return {
		arg
		for arg in sp.preorder_traversal(expression)
		if isinstance(arg, spu.Quantity)
	}

####################
# - Force
####################
# Newton
nanonewton = nN = spu.Quantity("nanonewton", abbrev="nN")
nanonewton.set_global_relative_scale_factor(spu.nano, spu.newton)

micronewton = uN = spu.Quantity("micronewton", abbrev="μN")
micronewton.set_global_relative_scale_factor(spu.micro, spu.newton)

millinewton = mN = spu.Quantity("micronewton", abbrev="mN")
micronewton.set_global_relative_scale_factor(spu.milli, spu.newton)

####################
# - Frequency
####################
# Hertz
kilohertz = kHz = spu.Quantity("kilohertz", abbrev="kHz")
kilohertz.set_global_relative_scale_factor(spu.kilo, spu.hertz)

megahertz = MHz = spu.Quantity("megahertz", abbrev="MHz")
kilohertz.set_global_relative_scale_factor(spu.kilo, spu.hertz)

gigahertz = GHz = spu.Quantity("gigahertz", abbrev="GHz")
gigahertz.set_global_relative_scale_factor(spu.giga, spu.hertz)

terahertz = THz = spu.Quantity("terahertz", abbrev="THz")
terahertz.set_global_relative_scale_factor(spu.tera, spu.hertz)

petahertz = PHz = spu.Quantity("petahertz", abbrev="PHz")
petahertz.set_global_relative_scale_factor(spu.peta, spu.hertz)

exahertz = EHz = spu.Quantity("exahertz", abbrev="EHz")
exahertz.set_global_relative_scale_factor(spu.exa, spu.hertz)

####################
# - Sympy Expression Typing
####################
#ALL_UNIT_SYMBOLS = {
#	unit
#	for unit in spu.__dict__.values()
#	if isinstance(unit, spu.Quantity)
#}
#def has_units(expr: sp.Expr):
#	return any(
#		symbol in ALL_UNIT_SYMBOLS
#		for symbol in expr.atoms(sp.Symbol)
#	)
#def is_exactly_expressed_as_unit(expr: sp.Expr, unit) -> bool:
#	#try:
#	converted_expr = expr / unit
#	
#	return (
#		converted_expr.is_number
#		and not converted_expr.has(spu.Quantity)
#	)
