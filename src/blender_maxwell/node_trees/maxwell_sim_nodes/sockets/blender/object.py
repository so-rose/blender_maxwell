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

log = logger.get(__name__)


####################
# - Blender Socket
####################
class BlenderObjectBLSocket(base.MaxwellSimSocket):
	socket_type = ct.SocketType.BlenderObject
	bl_label = 'Blender Object'

	####################
	# - Properties
	####################
	raw_value: bpy.types.Object = bl_cache.BLField()

	####################
	# - UI
	####################
	def draw_value(self, col: bpy.types.UILayout) -> None:
		col.prop(self, 'raw_value', text='')

	####################
	# - Default Value
	####################
	@bl_cache.cached_bl_property(depends_on={'raw_value'})
	def value(self) -> bpy.types.Object | None:
		return self.raw_value

	@value.setter
	def value(self, value: bpy.types.Object) -> None:
		self.raw_value = value


####################
# - Socket Configuration
####################
class BlenderObjectSocketDef(base.SocketDef):
	socket_type: ct.SocketType = ct.SocketType.BlenderObject

	def init(self, bl_socket: BlenderObjectBLSocket) -> None:
		pass


####################
# - Blender Registration
####################
BL_REGISTER = [
	BlenderObjectBLSocket,
]
