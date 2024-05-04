# blender_maxwell
# Copyright (C) 2024 blender_maxwell Project Contributors
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

from . import any as any_socket
from . import bool as bool_socket
from . import file_path, string

AnySocketDef = any_socket.AnySocketDef
BoolSocketDef = bool_socket.BoolSocketDef
StringSocketDef = string.StringSocketDef
FilePathSocketDef = file_path.FilePathSocketDef


BL_REGISTER = [
	*any_socket.BL_REGISTER,
	*bool_socket.BL_REGISTER,
	*string.BL_REGISTER,
	*file_path.BL_REGISTER,
]
