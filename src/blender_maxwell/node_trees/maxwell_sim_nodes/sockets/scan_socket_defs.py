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

import types

from .. import contracts as ct


## TODO: Replace with BL_SOCKET_DEFS export from each module, a little like BL_NODES.
def scan_for_socket_defs(
	sockets_module: types.ModuleType,
) -> dict:
	return {
		socket_type: getattr(
			sockets_module,
			socket_type.value.removesuffix('SocketType') + 'SocketDef',
		)
		for socket_type in ct.SocketType
		if hasattr(
			sockets_module, socket_type.value.removesuffix('SocketType') + 'SocketDef'
		)
	}


## TODO: Function for globals() filling too.
