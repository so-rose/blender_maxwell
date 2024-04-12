"""A managed Blender modifier, associated with some Blender object."""

import typing as typ

import bpy
import typing_extensions as typx

from ....utils import analyze_geonodes, logger
from .. import bl_socket_map
from .. import contracts as ct

log = logger.get(__name__)

ModifierType: typ.TypeAlias = typx.Literal['NODES', 'ARRAY']
NodeTreeInterfaceID: typ.TypeAlias = str
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
		inputs: Values to associate with each GeoNodes interface socket.
			Use `analyze_geonodes.interface(..., direc='INPUT')` to determine acceptable values.
	"""

	node_group: bpy.types.GeometryNodeTree
	unit_system: UnitSystem
	inputs: dict[NodeTreeInterfaceID, typ.Any]


class ModifierAttrsARRAY(typ.TypedDict):
	"""Describes values set on an Array modifier."""


ModifierAttrs: typ.TypeAlias = ModifierAttrsNODES | ModifierAttrsARRAY

MODIFIER_NAMES = {
	'NODES': 'BLMaxwell_GeoNodes',
	'ARRAY': 'BLMaxwell_Array',
}


####################
# - Read Modifier Information
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
# - Write Modifier Information
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
	## First we retrieve the interface items by-Socket Name
	geonodes_interface = analyze_geonodes.interface(
		bl_modifier.node_group, direc='INPUT'
	)
	for (
		socket_name,
		value,
	) in modifier_attrs['inputs'].items():
		# Compute Writable BL Socket Value
		## Analyzes the socket and unitsys to prep a ready-to-write value.
		## Write directly to the modifier dict.
		bl_socket_value = bl_socket_map.writable_bl_socket_value(
			geonodes_interface[socket_name],
			value,
			unit_system=modifier_attrs['unit_system'],
			allow_unit_not_in_unit_system=True,
		)

		# Compute Interface ID from Socket Name
		## We can't index the modifier by socket name; only by Interface ID.
		## Still, we require that socket names are unique.
		iface_id = geonodes_interface[socket_name].identifier

		# IF List-Like: Alter Differing Elements
		if isinstance(bl_socket_value, tuple):
			for i, bl_socket_subvalue in enumerate(bl_socket_value):
				if bl_modifier[iface_id][i] != bl_socket_subvalue:
					bl_modifier[iface_id][i] = bl_socket_subvalue
					modifier_altered = True

		# IF int/float Mismatch: Assign Float-Cast of Integer
		## Blender is strict; only floats can set float vals.
		## We are less strict; if the user passes an int, that's okay.
		elif isinstance(bl_socket_value, int) and isinstance(
			bl_modifier[iface_id],
			float,
		):
			bl_modifier[iface_id] = float(bl_socket_value)
			modifier_altered = True
		else:
			## TODO: Whitelist what can be here. I'm done with the TypeErrors.
			bl_modifier[iface_id] = bl_socket_value
			modifier_altered = True

	return modifier_altered


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
class ManagedBLModifier(ct.schemas.ManagedObj):
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
		modifier_type: ModifierType,
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

		if modifier_altered := write_modifier(bl_modifier, modifier_attrs):
			bl_object.data.update()

		return bl_modifier
