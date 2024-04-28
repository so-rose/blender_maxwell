from . import pol, unit_system

PhysicalPolSocketDef = pol.PhysicalPolSocketDef


BL_REGISTER = [
	*unit_system.BL_REGISTER,
	*pol.BL_REGISTER,
]
