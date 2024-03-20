from . import any as any_socket

AnySocketDef = any_socket.AnySocketDef

from . import bool as bool_socket

BoolSocketDef = bool_socket.BoolSocketDef

from . import string

StringSocketDef = string.StringSocketDef

from . import file_path

FilePathSocketDef = file_path.FilePathSocketDef


BL_REGISTER = [
	*any_socket.BL_REGISTER,
	*bool_socket.BL_REGISTER,
	*string.BL_REGISTER,
	*file_path.BL_REGISTER,
]
