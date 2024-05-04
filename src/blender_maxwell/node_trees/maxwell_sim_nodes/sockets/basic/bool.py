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
	raw_value: bpy.props.BoolProperty(
		name='Boolean',
		description='Represents a boolean value',
		default=False,
		update=(lambda self, context: self.on_prop_changed('raw_value', context)),
	)

	####################
	# - Socket UI
	####################
	def draw_label_row(self, label_col_row: bpy.types.UILayout, text: str) -> None:
		label_col_row.label(text=text)
		label_col_row.prop(self, 'raw_value', text='')

	####################
	# - Computation of Default Value
	####################
	@property
	def value(self) -> bool:
		return self.raw_value

	@value.setter
	def value(self, value: bool) -> None:
		self.raw_value = value


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
