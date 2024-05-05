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

"""A managed Blender modifier, associated with some Blender object."""

import typing as typ

import bpy

from blender_maxwell.utils import extra_sympy_units as spux
from blender_maxwell.utils import logger

from .. import bl_socket_map
from .. import contracts as ct
from . import base

log = logger.get(__name__)

UnitSystem: typ.TypeAlias = typ.Any


####################
# - Modifier Attributes
####################
class ModifierAttrsNODES(typ.TypedDict):
	"""Describes values set on an GeoNodes modifier.

	Attributes:
		node_group: The GeoNodes group to use in the modifier.
		unit_system: The unit system used by the GeoNodes output.
			Generally, `ct.UNITS_BLENDER` is a good choice.
		inputs: Values to associate with each GeoNodes interface socket name.
	"""

	node_group: bpy.types.GeometryNodeTree
	unit_system: UnitSystem
	inputs: dict[ct.SocketName, typ.Any]


class ModifierAttrsARRAY(typ.TypedDict):
	"""Describes values set on an Array modifier."""


ModifierAttrs: typ.TypeAlias = ModifierAttrsNODES | ModifierAttrsARRAY

MODIFIER_NAMES = {
	'NODES': 'BLMaxwell_GeoNodes',
	'ARRAY': 'BLMaxwell_Array',
}


####################
# - Read Modifier
####################
def read_modifier(bl_modifier: bpy.types.Modifier) -> ModifierAttrs:
	if bl_modifier.type == 'NODES':
		## TODO: Also get GeoNodes modifier values, if the nodegroup is not-None.
		return {
			'node_group': bl_modifier.node_group,
		}
	elif bl_modifier.type == 'ARRAY':
		raise NotImplementedError

	raise NotImplementedError


####################
# - Write Modifier: GeoNodes
####################
def write_modifier_geonodes(
	bl_modifier: bpy.types.NodesModifier,
	modifier_attrs: ModifierAttrsNODES,
) -> bool:
	"""Writes attributes to the GeoNodes modifier, changing only what's needed.

	Parameters:
		bl_modifier: The GeoNodes modifier to write to.
		modifier_attrs: The attributes to write to

	Returns:
		True if the modifier was altered.
	"""
	modifier_altered = False

	# Alter GeoNodes Group
	if bl_modifier.node_group != modifier_attrs['node_group']:
		log.info(
			'Changing GeoNodes Modifier NodeTree from "%s" to "%s"',
			str(bl_modifier.node_group),
			str(modifier_attrs['node_group']),
		)
		bl_modifier.node_group = modifier_attrs['node_group']
		modifier_altered = True

	# Alter GeoNodes Modifier Inputs
	socket_infos = bl_socket_map.info_from_geonodes(bl_modifier.node_group)

	for socket_name in modifier_attrs['inputs']:
		iface_id = socket_infos[socket_name].bl_isocket_identifier
		input_value = modifier_attrs['inputs'][socket_name]

		if isinstance(input_value, spux.SympyType):
			bl_modifier[iface_id] = spux.scale_to_unit_system(
				input_value, modifier_attrs['unit_system']
			)
		else:
			bl_modifier[iface_id] = input_value

	modifier_altered = True
	## TODO: More fine-grained alterations

	return modifier_altered  # noqa: RET504


####################
# - Write Modifier
####################
def write_modifier(
	bl_modifier: bpy.types.Modifier,
	modifier_attrs: ModifierAttrs,
) -> bool:
	"""Writes modifier attributes to the modifier, changing only what's needed.

	Returns:
		True if the modifier was altered.
	"""
	modifier_altered = False
	if bl_modifier.type == 'NODES':
		modifier_altered = write_modifier_geonodes(bl_modifier, modifier_attrs)
	elif bl_modifier.type == 'ARRAY':
		raise NotImplementedError
	else:
		raise NotImplementedError

	return modifier_altered


####################
# - ManagedObj
####################
class ManagedBLModifier(base.ManagedObj):
	managed_obj_type = ct.ManagedObjType.ManagedBLModifier
	_modifier_name: str | None = None

	####################
	# - BL Object Name
	####################
	@property
	def name(self):
		return self._modifier_name

	@name.setter
	def name(self, value: str) -> None:
		## TODO: Handle name conflict within same BLObject
		log.info(
			'Changing BLModifier w/Name "%s" to Name "%s"', self._modifier_name, value
		)
		self._modifier_name = value

	####################
	# - Allocation
	####################
	def __init__(self, name: str):
		self.name = name

	def bl_select(self) -> None:
		pass

	def hide_preview(self) -> None:
		pass

	####################
	# - Deallocation
	####################
	def free(self):
		"""Not needed - when the object is removed, its modifiers are also removed."""

	def free_from_bl_object(
		self,
		bl_object: bpy.types.Object,
	) -> None:
		"""Remove the managed BL modifier from the passed Blender object.

		Parameters:
			bl_object: The Blender object to remove the modifier from.
		"""
		if (bl_modifier := bl_object.modifiers.get(self.name)) is not None:
			log.info(
				'Removing (recreating) BLModifier "%s" on BLObject "%s" (existing modifier_type is "%s")',
				bl_modifier.name,
				bl_object.name,
				bl_modifier.type,
			)
			bl_modifier = bl_object.modifiers.remove(bl_modifier)
		else:
			msg = f'Tried to free bl_modifier "{self.name}", but bl_object "{bl_object.name}" has no modifier of that name'
			raise ValueError(msg)

	####################
	# - Modifiers
	####################
	def bl_modifier(
		self,
		bl_object: bpy.types.Object,
		modifier_type: ct.BLModifierType,
		modifier_attrs: ModifierAttrs,
	):
		"""Creates a new modifier for the current `bl_object`.

		- Modifier Type Names: <https://docs.blender.org/api/current/bpy_types_enum_items/object_modifier_type_items.html#rna-enum-object-modifier-type-items>
		"""
		# Remove Mismatching Modifier
		modifier_was_removed = False
		if (
			bl_modifier := bl_object.modifiers.get(self.name)
		) and bl_modifier.type != modifier_type:
			log.info(
				'Removing (recreating) BLModifier "%s" on BLObject "%s" (existing modifier_type is "%s", but "%s" is requested)',
				bl_modifier.name,
				bl_object.name,
				bl_modifier.type,
				modifier_type,
			)
			self.free_from_bl_object(bl_object)
			modifier_was_removed = True

		# Create Modifier
		if bl_modifier is None or modifier_was_removed:
			log.info(
				'Creating BLModifier "%s" on BLObject "%s" with modifier_type "%s"',
				self.name,
				bl_object.name,
				modifier_type,
			)
			bl_modifier = bl_object.modifiers.new(
				name=self.name,
				type=modifier_type,
			)

		modifier_altered = write_modifier(bl_modifier, modifier_attrs)
		if modifier_altered:
			bl_object.data.update()

		return bl_modifier
