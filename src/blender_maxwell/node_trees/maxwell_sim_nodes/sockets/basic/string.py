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

import bpy

from blender_maxwell.utils import bl_cache

from ... import contracts as ct
from .. import base


####################
# - Blender Socket
####################
class StringBLSocket(base.MaxwellSimSocket):
	socket_type = ct.SocketType.String
	bl_label = 'String'

	####################
	# - Properties
	####################
	raw_value: str = bl_cache.BLField('')

	####################
	# - Socket UI
	####################
	def draw_label_row(self, label_col_row: bpy.types.UILayout, text: str) -> None:
		label_col_row.prop(self, 'raw_value', text=text)

	####################
	# - Computation of Default Value
	####################
	@bl_cache.cached_bl_property(depends_on={'raw_value'})
	def value(self) -> str:
		return self.raw_value

	@value.setter
	def value(self, value: str) -> None:
		self.raw_value = value


####################
# - Socket Configuration
####################
class StringSocketDef(base.SocketDef):
	socket_type: ct.SocketType = ct.SocketType.String

	default_text: str = ''

	def init(self, bl_socket: StringBLSocket) -> None:
		bl_socket.value = self.default_text


####################
# - Blender Registration
####################
BL_REGISTER = [
	StringBLSocket,
]
