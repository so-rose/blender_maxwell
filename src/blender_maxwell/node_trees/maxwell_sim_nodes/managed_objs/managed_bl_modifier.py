import typing as typ
import bpy
import typing_extensions as typx

from ....utils import analyze_geonodes
from ....utils import logger
from .. import contracts as ct

log = logger.get(__name__)

ModifierType: typ.TypeAlias = typx.Literal['NODES', 'ARRAY']


NodeTreeInterfaceID: typ.TypeAlias = str


class ModifierAttrsNODES(typ.TypedDict):
	node_group: bpy.types.GeometryNodeTree
	inputs: dict[NodeTreeInterfaceID, typ.Any]


class ModifierAttrsARRAY(typ.TypedDict):
	pass


ModifierAttrs: typ.TypeAlias = ModifierAttrsNODES | ModifierAttrsARRAY
MODIFIER_NAMES = {
	'NODES': 'BLMaxwell_GeoNodes',
	'ARRAY': 'BLMaxwell_Array',
}


####################
# - Read/Write Modifier Attributes
####################
def read_modifier(bl_modifier: bpy.types.Modifier) -> ModifierAttrs:
	if bl_modifier.type == 'NODES':
		return {
			'node_group': bl_modifier.node_group,
		}
	elif bl_modifier.type == 'ARRAY':
		raise NotImplementedError

	raise NotImplementedError


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
		# Alter GeoNodes Group
		if bl_modifier.node_group != modifier_attrs['node_group']:
			log.info(
				'Changing GeoNodes Modifier NodeTree from "%s" to "%s"',
				str(bl_modifier.node_group),
				str(modifier_attrs['node_group']),
			)
			bl_modifier.node_group = modifier_attrs['node_group']
			modifier_altered = True

		# Alter GeoNodes Input (Interface) Socket Values
		## The modifier's dict-like setter actually sets NodeTree interface vals
		## By setting the interface value, this particular NodeTree will change
		geonodes_interface = analyze_geonodes.interface(
			bl_modifier.node_group, direct='INPUT'
		)
		for (
			socket_name,
			raw_value,
		) in modifier_attrs['inputs'].items():
			iface_id = geonodes_interface[socket_name].identifier
			# Alter Interface Value
			if bl_modifier[iface_id] != raw_value:
				# Determine IDPropertyArray Equality
				## The equality above doesn't work for IDPropertyArrays.
				## BUT, IDPropertyArrays must have a 'to_list' method.
				## To do the comparison, we tuple-ify the IDPropertyArray.
				## raw_value is always a tuple if it's listy.
				if (
					hasattr(bl_modifier[iface_id], 'to_list')
					and tuple(bl_modifier[iface_id].to_list()) == raw_value
				):
					continue

				# Determine int/float Mismatch
				## Blender is strict; only floats can set float vals.
				## We are less strict; if the user passes an int, that's okay.
				if isinstance(
					bl_modifier[iface_id],
					float,
				) and isinstance(raw_value, int):
					value = float(raw_value)

				bl_modifier[iface_id] = value
				modifier_altered = True
				## TODO: Altering existing values is much better for performance.
				## - GC churn is real!
				## - Especially since this is in a hot path

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
			'Changing BLModifier w/Name "%s" to Name "%s"', self._bl_object_name, value
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
		log.info('Freeing BLModifier w/Name "%s" (NOT IMPLEMENTED)', self.name)
		## TODO: Implement

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
			self.free()

		# Create Modifier
		if not (bl_modifier := bl_object.modifiers.get(self.name)):
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
