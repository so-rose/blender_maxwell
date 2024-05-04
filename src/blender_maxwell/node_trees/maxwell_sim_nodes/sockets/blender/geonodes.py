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
# - Operators
####################
class BlenderMaxwellResetGeoNodesSocket(bpy.types.Operator):
	bl_idname = 'blender_maxwell.reset_geo_nodes_socket'
	bl_label = 'Reset GeoNodes Socket'

	node_tree_name: bpy.props.StringProperty(name='Node Tree Name')
	node_name: bpy.props.StringProperty(name='Node Name')
	socket_name: bpy.props.StringProperty(name='Socket Name')

	def execute(self, context):
		node_tree = bpy.data.node_groups[self.node_tree_name]
		node = node_tree.nodes[self.node_name]
		socket = node.inputs[self.socket_name]

		# Report as though the GeoNodes Tree Changed
		socket.on_prop_changed('raw_value', context)

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
	raw_value: bpy.props.PointerProperty(
		name='Blender GeoNodes Tree',
		description='Represents a Blender GeoNodes Tree',
		type=bpy.types.NodeTree,
		poll=(lambda self, obj: obj.bl_idname == 'GeometryNodeTree'),
		update=(lambda self, context: self.on_prop_changed('raw_value', context)),
	)

	####################
	# - UI
	####################
	# def draw_label_row(self, label_col_row, text):
	# label_col_row.label(text=text)
	# if not self.raw_value: return
	#
	# op = label_col_row.operator(
	# BlenderMaxwellResetGeoNodesSocket.bl_idname,
	# text="",
	# icon="FILE_REFRESH",
	# )
	# op.socket_name = self.name
	# op.node_name = self.node.name
	# op.node_tree_name = self.node.id_data.name

	####################
	# - UI
	####################
	def draw_value(self, col: bpy.types.UILayout) -> None:
		col.prop(self, 'raw_value', text='')

	####################
	# - Default Value
	####################
	@property
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
