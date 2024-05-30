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
class BoolBLSocket(base.MaxwellSimSocket):
	socket_type = ct.SocketType.Bool
	bl_label = 'Bool'

	####################
	# - Properties
	####################
	raw_value: bool = bl_cache.BLField(False)

	####################
	# - Socket UI
	####################
	def draw_label_row(self, label_col_row: bpy.types.UILayout, text: str) -> None:
		label_col_row.prop(self, self.blfields['raw_value'], text=text, toggle=True)

	####################
	# - Computation of Default Value
	####################
	@bl_cache.cached_bl_property(depends_on={'raw_value'})
	def value(self) -> bool:
		return self.raw_value

	@value.setter
	def value(self, value: bool) -> None:
		self.raw_value = value

	@bl_cache.cached_bl_property(depends_on={'value'})
	def lazy_func(self) -> ct.FuncFlow:
		return ct.FuncFlow(
			func=lambda: self.value,
		)

	@bl_cache.cached_bl_property()
	def params(self) -> ct.FuncFlow:
		return ct.ParamsFlow()


####################
# - Socket Configuration
####################
class BoolSocketDef(base.SocketDef):
	socket_type: ct.SocketType = ct.SocketType.Bool

	default_value: bool = False

	def init(self, bl_socket: BoolBLSocket) -> None:
		bl_socket.value = self.default_value


####################
# - Blender Registration
####################
BL_REGISTER = [
	BoolBLSocket,
]
