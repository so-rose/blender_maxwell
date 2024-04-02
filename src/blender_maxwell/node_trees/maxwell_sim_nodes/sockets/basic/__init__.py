from . import any as any_socket
from . import bool as bool_socket
from . import file_path, string

AnySocketDef = any_socket.AnySocketDef
BoolSocketDef = bool_socket.BoolSocketDef
FilePathSocketDef = file_path.FilePathSocketDef
StringSocketDef = string.StringSocketDef


BL_REGISTER = [
	*any_socket.BL_REGISTER,
	*bool_socket.BL_REGISTER,
	*string.BL_REGISTER,
	*file_path.BL_REGISTER,
]
