from . import real_2d_vector_socket
from . import complex_2d_vector_socket
Real2DVectorSocketDef = real_2d_vector_socket.Real2DVectorSocketDef
Complex2DVectorSocketDef = complex_2d_vector_socket.Complex2DVectorSocketDef

from . import real_3d_vector_socket
from . import complex_3d_vector_socket
Real3DVectorSocketDef = real_3d_vector_socket.Real3DVectorSocketDef
Complex3DVectorSocketDef = complex_3d_vector_socket.Complex3DVectorSocketDef


BL_REGISTER = [
	*real_2d_vector_socket.BL_REGISTER,
	*complex_2d_vector_socket.BL_REGISTER,
	
	*real_3d_vector_socket.BL_REGISTER,
	*complex_3d_vector_socket.BL_REGISTER,
]
