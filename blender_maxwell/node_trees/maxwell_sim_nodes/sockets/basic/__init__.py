from . import any_socket
AnySocketDef = any_socket.AnySocketDef

from . import bool_socket
BoolSocketDef = bool_socket.BoolSocketDef

from . import text_socket
TextSocketDef = text_socket.TextSocketDef

from . import file_path_socket
FilePathSocketDef = file_path_socket.FilePathSocketDef


BL_REGISTER = [
	*any_socket.BL_REGISTER,
	*bool_socket.BL_REGISTER,
	*text_socket.BL_REGISTER,
	*file_path_socket.BL_REGISTER,
]
