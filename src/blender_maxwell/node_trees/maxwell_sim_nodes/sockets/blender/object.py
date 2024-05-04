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
# - Create and Assign BL Object
####################
class BlenderMaxwellCreateAndAssignBLObject(bpy.types.Operator):
	bl_idname = 'blender_maxwell.create_and_assign_bl_object'
	bl_label = 'Create and Assign BL Object'

	node_tree_name = bpy.props.StringProperty(name='Node Tree Name')
	node_name = bpy.props.StringProperty(name='Node Name')
	socket_name = bpy.props.StringProperty(name='Socket Name')

	def execute(self, context):
		node_tree = bpy.data.node_groups[self.node_tree_name]
		node = node_tree.nodes[self.node_name]
		socket = node.inputs[self.socket_name]

		socket.create_and_assign_bl_object()

		return {'FINISHED'}


####################
# - Blender Socket
####################
class BlenderObjectBLSocket(base.MaxwellSimSocket):
	socket_type = ct.SocketType.BlenderObject
	bl_label = 'Blender Object'

	####################
	# - Properties
	####################
	raw_value: bpy.props.PointerProperty(
		name='Blender Object',
		description='Represents a Blender object',
		type=bpy.types.Object,
		update=(lambda self, context: self.on_prop_changed('raw_value', context)),
	)

	####################
	# - UI
	####################
	def draw_label_row(self, label_col_row, text):
		label_col_row.label(text=text)

		op = label_col_row.operator(
			'blender_maxwell.create_and_assign_bl_object',
			text='',
			icon='ADD',
		)
		op.socket_name = self.name
		op.node_name = self.node.name
		op.node_tree_name = self.node.id_data.name

	def draw_value(self, col: bpy.types.UILayout) -> None:
		col.prop(self, 'raw_value', text='')

	####################
	# - Methods
	####################
	def create_and_assign_bl_object(self):
		node_tree = self.node.id_data
		mesh = bpy.data.meshes.new('MaxwellMesh')
		new_bl_object = bpy.data.objects.new('MaxwellObject', mesh)

		bpy.context.collection.objects.link(new_bl_object)

		self.value = new_bl_object

	####################
	# - Default Value
	####################
	@property
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
	BlenderMaxwellCreateAndAssignBLObject,
	BlenderObjectBLSocket,
]
