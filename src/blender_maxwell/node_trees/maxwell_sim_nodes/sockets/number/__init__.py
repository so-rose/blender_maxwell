from . import integer_number
IntegerNumberSocketDef = integer_number.IntegerNumberSocketDef

from . import rational_number
RationalNumberSocketDef = rational_number.RationalNumberSocketDef

from . import real_number
RealNumberSocketDef = real_number.RealNumberSocketDef

from . import complex_number
ComplexNumberSocketDef = complex_number.ComplexNumberSocketDef


BL_REGISTER = [
	*integer_number.BL_REGISTER,
	*rational_number.BL_REGISTER,
	*real_number.BL_REGISTER,
	*complex_number.BL_REGISTER,
]
