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

from ... import contracts as ct
from .. import base


class MaxwellMonitorBLSocket(base.MaxwellSimSocket):
	socket_type = ct.SocketType.MaxwellMonitor
	bl_label = 'Maxwell Monitor'


####################
# - Socket Configuration
####################
class MaxwellMonitorSocketDef(base.SocketDef):
	socket_type: ct.SocketType = ct.SocketType.MaxwellMonitor

	is_list: bool = False

	def init(self, bl_socket: MaxwellMonitorBLSocket) -> None:
		if self.is_list:
			bl_socket.active_kind = ct.FlowKind.Array


####################
# - Blender Registration
####################
BL_REGISTER = [
	MaxwellMonitorBLSocket,
]
