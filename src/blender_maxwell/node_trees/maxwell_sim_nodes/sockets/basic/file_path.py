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

from pathlib import Path

import bpy

from ... import contracts as ct
from .. import base


####################
# - Blender Socket
####################
class FilePathBLSocket(base.MaxwellSimSocket):
	socket_type = ct.SocketType.FilePath
	bl_label = 'File Path'

	####################
	# - Properties
	####################
	raw_value: bpy.props.StringProperty(
		name='File Path',
		description='Represents the path to a file',
		subtype='FILE_PATH',
		update=(lambda self, context: self.on_prop_changed('raw_value', context)),
	)

	####################
	# - Socket UI
	####################
	def draw_value(self, col: bpy.types.UILayout) -> None:
		col_row = col.row(align=True)
		col_row.prop(self, 'raw_value', text='')

	####################
	# - Computation of Default Value
	####################
	@property
	def value(self) -> Path:
		return Path(bpy.path.abspath(self.raw_value))

	@value.setter
	def value(self, value: Path) -> None:
		self.raw_value = bpy.path.relpath(str(value))


####################
# - Socket Configuration
####################
class FilePathSocketDef(base.SocketDef):
	socket_type: ct.SocketType = ct.SocketType.FilePath

	default_path: Path = Path()

	def init(self, bl_socket: FilePathBLSocket) -> None:
		bl_socket.value = self.default_path


####################
# - Blender Registration
####################
BL_REGISTER = [
	FilePathBLSocket,
]
