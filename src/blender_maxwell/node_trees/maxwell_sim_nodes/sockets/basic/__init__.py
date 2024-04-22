from . import any as any_socket
from . import bool as bool_socket
from . import expr, file_path, string, data

AnySocketDef = any_socket.AnySocketDef
DataSocketDef = data.DataSocketDef
BoolSocketDef = bool_socket.BoolSocketDef
StringSocketDef = string.StringSocketDef
FilePathSocketDef = file_path.FilePathSocketDef
ExprSocketDef = expr.ExprSocketDef


BL_REGISTER = [
	*any_socket.BL_REGISTER,
	*data.BL_REGISTER,
	*bool_socket.BL_REGISTER,
	*string.BL_REGISTER,
	*file_path.BL_REGISTER,
	*expr.BL_REGISTER,
]
