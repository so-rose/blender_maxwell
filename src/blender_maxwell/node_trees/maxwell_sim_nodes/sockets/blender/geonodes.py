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
# - Operators
####################
class BlenderMaxwellResetGeoNodesSocket(bpy.types.Operator):
	"""Simulate a change to the geometry nodes group of the attached `GeoNodes` socket.

	This causes updates to the GN group (ex. internal logic) to be immediately caught.
	"""

	bl_idname = ct.OperatorType.SocketGeoNodesReset
	bl_label = 'Reset GeoNodes Group'

	def execute(self, context):
		bl_socket = context.socket

		# Report as though the GeoNodes Tree Changed
		bl_socket.on_prop_changed('raw_value', context)

		return {'FINISHED'}


####################
# - Blender Socket
####################
class BlenderGeoNodesBLSocket(base.MaxwellSimSocket):
	socket_type = ct.SocketType.BlenderGeoNodes
	bl_label = 'Geometry Node Tree'

	####################
	# - Properties
	####################
	raw_value: bpy.types.NodeTree = bl_cache.BLField(
		bltype_poll=lambda self, obj: self.filter_gn_trees(obj)
	)

	def filter_gn_trees(self, obj: ct.BLIDStruct) -> bool:
		return obj.bl_idname == 'GeometryNodeTree' and not obj.name.startswith('_')

	####################
	# - UI
	####################
	def draw_value(self, col: bpy.types.UILayout) -> None:
		col.prop(self, self.blfields['raw_value'], text='')

	####################
	# - Default Value
	####################
	@bl_cache.cached_bl_property(depends_on={'raw_value'})
	def value(self) -> bpy.types.NodeTree | ct.FlowSignal:
		return self.raw_value if self.raw_value is not None else ct.FlowSignal.NoFlow

	@value.setter
	def value(self, value: bpy.types.NodeTree) -> None:
		self.raw_value = value


####################
# - Socket Configuration
####################
class BlenderGeoNodesSocketDef(base.SocketDef):
	socket_type: ct.SocketType = ct.SocketType.BlenderGeoNodes

	def init(self, bl_socket: BlenderGeoNodesBLSocket) -> None:
		pass


####################
# - Blender Registration
####################
BL_REGISTER = [
	BlenderMaxwellResetGeoNodesSocket,
	BlenderGeoNodesBLSocket,
]
