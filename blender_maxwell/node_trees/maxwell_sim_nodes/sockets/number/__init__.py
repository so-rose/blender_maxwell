from . import integer_number_socket
IntegerNumberSocketDef = integer_number_socket.IntegerNumberSocketDef

from . import rational_number_socket
RationalNumberSocketDef = rational_number_socket.RationalNumberSocketDef

from . import real_number_socket
RealNumberSocketDef = real_number_socket.RealNumberSocketDef

from . import complex_number_socket
ComplexNumberSocketDef = complex_number_socket.ComplexNumberSocketDef


BL_REGISTER = [
	*integer_number_socket.BL_REGISTER,
	*rational_number_socket.BL_REGISTER,
	*real_number_socket.BL_REGISTER,
	*complex_number_socket.BL_REGISTER,
]
