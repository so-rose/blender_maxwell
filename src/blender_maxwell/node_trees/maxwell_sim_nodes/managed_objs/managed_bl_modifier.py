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
import jax
import numpy as np

from blender_maxwell.utils import logger

from .. import bl_socket_map
from .. import contracts as ct
from . import base
from .managed_bl_mesh import ManagedBLMesh

log = logger.get(__name__)

UnitSystem: typ.TypeAlias = typ.Any


####################
# - Modifier Attributes
####################
class ModifierAttrsNODES(typ.TypedDict):
	"""Describes values set on an GeoNodes modifier.

	Attributes:
		node_group: The GeoNodes group to use in the modifier.
		inputs: Values to associate with each GeoNodes interface socket name.
	"""

	node_group: bpy.types.GeometryNodeTree
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
	if bl_modifier.type == 'ARRAY':
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
	## -> Check the existing node group, replace if it differs.
	if bl_modifier.node_group != modifier_attrs['node_group']:
		log.info(
			'Changing GeoNodes Modifier NodeTree from "%s" to "%s"',
			str(bl_modifier.node_group),
			str(modifier_attrs['node_group']),
		)
		bl_modifier.node_group = modifier_attrs['node_group']
		modifier_altered = True

	# Parse GeoNodes Socket Info
	## -> TODO: Slow and hard to optimize, but very likely worth it.
	socket_infos = bl_socket_map.info_from_geonodes(bl_modifier.node_group)

	for socket_name in modifier_attrs['inputs']:
		# Retrieve Modifier Interface ID
		## -> iface_id translates "modifier socket" to "GN input socket".
		iface_id = socket_infos[socket_name].bl_isocket_identifier

		# Deduce Value to Write
		## -> This may involve a unit system conversion.
		## -> Special Case: Booleans do not go through unit conversion.
		## -> TODO: A special case isn't clean enough.
		bl_modifier[iface_id] = socket_infos[socket_name].encode(
			raw_value=modifier_attrs['inputs'][socket_name],
		)
		modifier_altered = True
		## TODO: More fine-grained alterations?

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
	twin_bl_mesh: ManagedBLMesh | None = None

	####################
	# - BL Object Name
	####################
	@property
	def name(self):
		return self._modifier_name

	@name.setter
	def name(self, value: str) -> None:
		log.debug('Changing BLModifier w/Name "%s" to Name "%s"', self.name, value)

		twin_bl_object = bpy.data.objects.get(self.twin_bl_mesh.name)

		# No Existing Twin BLObject
		## -> Since no modifier-holding object exists, we're all set.
		if twin_bl_object is None:
			self._modifier_name = value

		# Existing Twin BLObject
		else:
			# No Existing Modifier: Set Value to Name
			## -> We'll rename the bl_object; otherwise we're set.
			bl_modifier = twin_bl_object.modifiers.get(self.name)
			if bl_modifier is None:
				self.twin_bl_mesh.name = value
				self._modifier_name = value

			# Existing Modifier: Rename to New Name
			## -> We'll rename the bl_modifier, then the bl_object.
			else:
				bl_modifier.name = value
				self.twin_bl_mesh.name = value
				self._modifier_name = value

	####################
	# - Allocation
	####################
	def __init__(self, name: str, prev_name: str | None = None):
		self.twin_bl_mesh = ManagedBLMesh(name, prev_name=prev_name)
		if prev_name is not None:
			self._modifier_name = prev_name
		else:
			self._modifier_name = name

		self.name = name

	def bl_select(self) -> None:
		self.twin_bl_mesh.bl_select()

	def show_preview(self) -> None:
		self.twin_bl_mesh.show_preview()

	def hide_preview(self) -> None:
		self.twin_bl_mesh.hide_preview()

	####################
	# - Deallocation
	####################
	def free(self):
		log.info('BLModifier: Freeing "%s" w/Twin BLObject of same name', self.name)
		self.twin_bl_mesh.free()

	####################
	# - Modifiers
	####################
	def bl_modifier(
		self,
		modifier_type: ct.BLModifierType,
		modifier_attrs: ModifierAttrs,
		location: np.ndarray | jax.Array | tuple[float, float, float] = (0, 0, 0),
	):
		"""Creates a new modifier for the current `bl_object`.

		- Modifier Type Names: <https://docs.blender.org/api/current/bpy_types_enum_items/object_modifier_type_items.html#rna-enum-object-modifier-type-items>
		"""
		# Retrieve Twin BLObject
		twin_bl_object = self.twin_bl_mesh.bl_object(location=location)
		if twin_bl_object is None:
			msg = f'BLModifier: No BLObject twin "{self.name}" exists to attach a modifier to.'
			raise ValueError(msg)

		bl_modifier = twin_bl_object.modifiers.get(self.name)

		# Existing Modifier: Maybe Remove
		modifier_was_removed = False
		if bl_modifier is not None and bl_modifier.type != modifier_type:
			log.info(
				'BLModifier: Clearing BLModifier "%s" from BLObject "%s"',
				self.name,
				twin_bl_object.name,
			)
			twin_bl_object.modifiers.remove(bl_modifier)
			modifier_was_removed = True

		# No/Removed Modifier: Create
		if bl_modifier is None or modifier_was_removed:
			log.info(
				'BLModifier: (Re)Creating BLModifier "%s" on BLObject "%s" (type=%s)',
				self.name,
				twin_bl_object.name,
				modifier_type,
			)
			bl_modifier = twin_bl_object.modifiers.new(
				name=self.name,
				type=modifier_type,
			)

		# Write Modifier Attrs
		## -> For GeoNodes modifiers, this is the critical component.
		## -> From 'write_modifier', we only need to know if something changed.
		## -> If so, we make sure to update the object data.
		modifier_altered = write_modifier(bl_modifier, modifier_attrs)
		if modifier_altered:
			twin_bl_object.data.update()

		return bl_modifier

	####################
	# - Mesh Data
	####################
	@property
	def mesh_as_arrays(self) -> dict:
		return self.twin_bl_mesh.mesh_as_arrays
