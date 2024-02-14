from . import real_number_socket
RealNumberSocketDef = real_number_socket.RealNumberSocketDef

from . import complex_number_socket
ComplexNumberSocketDef = complex_number_socket.ComplexNumberSocketDef


BL_REGISTER = [
	*real_number_socket.BL_REGISTER,
	*complex_number_socket.BL_REGISTER,
]
