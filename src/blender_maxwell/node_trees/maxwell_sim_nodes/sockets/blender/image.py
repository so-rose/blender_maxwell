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

from blender_maxwell.utils import bl_cache, logger

from ... import contracts as ct
from .. import base


####################
# - Blender Socket
####################
class BlenderImageBLSocket(base.MaxwellSimSocket):
	socket_type = ct.SocketType.BlenderImage
	bl_label = 'Blender Image'

	####################
	# - Properties
	####################
	raw_value: bpy.types.Image = bl_cache.BLField()

	####################
	# - UI
	####################
	def draw_value(self, col: bpy.types.UILayout) -> None:
		col.prop(self, 'raw_value', text='')

	####################
	# - Default Value
	####################
	@bl_cache.cached_bl_property(depends_on={'raw_value'})
	def value(self) -> bpy.types.Image | None:
		return self.raw_value

	@value.setter
	def value(self, value: bpy.types.Image) -> None:
		self.raw_value = value


####################
# - Socket Configuration
####################
class BlenderImageSocketDef(base.SocketDef):
	socket_type: ct.SocketType = ct.SocketType.BlenderImage

	def init(self, bl_socket: BlenderImageBLSocket) -> None:
		pass


####################
# - Blender Registration
####################
BL_REGISTER = [BlenderImageBLSocket]
