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


class BlenderMaterialBLSocket(base.MaxwellSimSocket):
	socket_type = ct.SocketType.BlenderMaterial
	bl_label = 'Blender Material'

	####################
	# - Properties
	####################
	raw_value: bpy.props.PointerProperty(
		name='Blender Material',
		description='Represents a Blender material',
		type=bpy.types.Material,
		update=(lambda self, context: self.on_prop_changed('raw_value', context)),
	)

	####################
	# - UI
	####################
	def draw_value(self, col: bpy.types.UILayout) -> None:
		col.prop(self, 'raw_value', text='')

	####################
	# - Default Value
	####################
	@property
	def value(self) -> bpy.types.Material | None:
		return self.raw_value

	@value.setter
	def value(self, value: bpy.types.Material) -> None:
		self.raw_value = value


####################
# - Socket Configuration
####################
class BlenderMaterialSocketDef(base.SocketDef):
	socket_type: ct.SocketType = ct.SocketType.BlenderMaterial

	def init(self, bl_socket: BlenderMaterialBLSocket) -> None:
		pass


####################
# - Blender Registration
####################
BL_REGISTER = [
	BlenderMaterialBLSocket,
]
