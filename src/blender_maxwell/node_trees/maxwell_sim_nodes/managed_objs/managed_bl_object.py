import contextlib

import bmesh
import bpy
import numpy as np
import typing_extensions as typx

from ....utils import logger
from .. import contracts as ct
from .managed_bl_collection import managed_collection, preview_collection

log = logger.get(__name__)

ModifierType = typx.Literal['NODES', 'ARRAY']
MODIFIER_NAMES = {
	'NODES': 'BLMaxwell_GeoNodes',
	'ARRAY': 'BLMaxwell_Array',
}


####################
# - BLObject
####################
class ManagedBLObject(ct.schemas.ManagedObj):
	managed_obj_type = ct.ManagedObjType.ManagedBLObject
	_bl_object_name: str | None = None

	####################
	# - BL Object Name
	####################
	@property
	def name(self):
		return self._bl_object_name

	@name.setter
	def name(self, value: str) -> None:
		log.info(
			'Changing BLObject w/Name "%s" to Name "%s"', self._bl_object_name, value
		)

		if not bpy.data.objects.get(value):
			log.info(
				'Desired BLObject Name "%s" Not Taken',
				value,
			)

			if self._bl_object_name is None:
				log.info(
					'Set New BLObject Name to "%s"',
					value,
				)
			elif bl_object := bpy.data.objects.get(self._bl_object_name):
				log.info(
					'Changed BLObject Name to "%s"',
					value,
				)
				bl_object.name = value
			else:
				msg = f'ManagedBLObject with name "{self._bl_object_name}" was deleted'
				raise RuntimeError(msg)

			# Set Internal Name
			self._bl_object_name = value
		else:
			log.info(
				'Desired BLObject Name "%s" is Taken. Using Blender Rename',
				value,
			)

			# Set Name Anyway, but Respect Blender's Renaming
			## When a name already exists, Blender adds .### to prevent overlap.
			## `set_name` is allowed to change the name; nodes account for this.
			bl_object.name = value
			self._bl_object_name = bl_object.name

			log.info(
				'Changed BLObject Name to "%s"',
				bl_object.name,
			)

	####################
	# - Allocation
	####################
	def __init__(self, name: str):
		self.name = name

	####################
	# - Deallocation
	####################
	def free(self):
		if (bl_object := bpy.data.objects.get(self.name)) is None:
			return

		# Delete the Underlying Datablock
		## This automatically deletes the object too
		log.info('Removing "%s" BLObject', bl_object.type)
		if bl_object.type in {'MESH', 'EMPTY'}:
			bpy.data.meshes.remove(bl_object.data)
		elif bl_object.type == 'VOLUME':
			bpy.data.volumes.remove(bl_object.data)
		else:
			msg = f'BLObject "{bl_object.name}" has invalid kind "{bl_object.type}"'
			raise RuntimeError(msg)

	####################
	# - Actions
	####################
	def show_preview(
		self,
		kind: typx.Literal['MESH', 'EMPTY', 'VOLUME'],
		empty_display_type: typx.Literal[
			'PLAIN_AXES',
			'ARROWS',
			'SINGLE_ARROW',
			'CIRCLE',
			'CUBE',
			'SPHERE',
			'CONE',
			'IMAGE',
		]
		| None = None,
	) -> None:
		"""Moves the managed Blender object to the preview collection.

		If it's already included, do nothing.
		"""
		bl_object = self.bl_object(kind)
		if bl_object.name not in preview_collection().objects:
			log.info('Moving "%s" to Preview Collection', bl_object.name)
			preview_collection().objects.link(bl_object)

		# Display Parameters
		if kind == 'EMPTY' and empty_display_type is not None:
			log.info(
				'Setting Empty Display Type "%s" for "%s"',
				empty_display_type,
				bl_object.name,
			)
			bl_object.empty_display_type = empty_display_type

	def hide_preview(
		self,
		kind: typx.Literal['MESH', 'EMPTY', 'VOLUME'],
	) -> None:
		"""Removes the managed Blender object from the preview collection.

		If it's already removed, do nothing.
		"""
		bl_object = self.bl_object(kind)
		if bl_object.name not in preview_collection().objects:
			log.info('Removing "%s" from Preview Collection', bl_object.name)
			preview_collection.objects.unlink(bl_object)

	def bl_select(self) -> None:
		"""Selects the managed Blender object globally, causing it to be ex.
		outlined in the 3D viewport.
		"""
		if (bl_object := bpy.data.objects.get(self.name)) is not None:
			bpy.ops.object.select_all(action='DESELECT')
			bl_object.select_set(True)

		msg = 'Managed BLObject does not exist'
		raise ValueError(msg)

	####################
	# - BLObject Management
	####################
	def bl_object(
		self,
		kind: typx.Literal['MESH', 'EMPTY', 'VOLUME'],
	):
		"""Returns the managed blender object.

		If the requested object data kind is different, then delete the old
		object and recreate.
		"""
		# Remove Object (if mismatch)
		if (bl_object := bpy.data.objects.get(self.name)) and bl_object.type != kind:
			log.info(
				'Removing (recreating) "%s" (existing kind is "%s", but "%s" is requested)',
				bl_object.name,
				bl_object.type,
				kind,
			)
			self.free()

		# Create Object w/Appropriate Data Block
		if not (bl_object := bpy.data.objects.get(self.name)):
			log.info(
				'Creating "%s" with kind "%s"',
				self.name,
				kind,
			)
			if kind == 'MESH':
				bl_data = bpy.data.meshes.new(self.name)
			elif kind == 'EMPTY':
				bl_data = None
			elif kind == 'VOLUME':
				raise NotImplementedError
			else:
				msg = f'Created BLObject w/invalid kind "{bl_object.type}" for "{self.name}"'
				raise ValueError(msg)

			bl_object = bpy.data.objects.new(self.name, bl_data)
			log.debug(
				'Linking "%s" to Base Collection',
				bl_object.name,
			)
			managed_collection().objects.link(bl_object)

		return bl_object

	####################
	# - Mesh Data Properties
	####################
	@property
	def mesh_data(self) -> bpy.types.Mesh:
		"""Directly loads the Blender mesh data.

		Raises:
			ValueError: If the object has no mesh data.
		"""
		if (bl_object := bpy.data.objects.get(self.name)) and bl_object.type == 'MESH':
			return bl_object.data

		msg = f'Requested mesh data from {self.name} of type {bl_object.type}'
		raise ValueError(msg)

	@contextlib.contextmanager
	def mesh_as_bmesh(
		self,
		evaluate: bool = True,
		triangulate: bool = False,
	) -> bpy.types.Mesh:
		if (bl_object := bpy.data.objects.get(self.name)) and bl_object.type == 'MESH':
			bmesh_mesh = None
			try:
				bmesh_mesh = bmesh.new()
				if evaluate:
					bmesh_mesh.from_object(
						bl_object,
						bpy.context.evaluated_depsgraph_get(),
					)
				else:
					bmesh_mesh.from_object(bl_object)

				if triangulate:
					bmesh.ops.triangulate(bmesh_mesh, faces=bmesh_mesh.faces)

				yield bmesh_mesh

			finally:
				if bmesh_mesh:
					bmesh_mesh.free()

		else:
			msg = f'Requested BMesh from "{self.name}" of type "{bl_object.type}"'
			raise ValueError(msg)

	@property
	def mesh_as_arrays(self) -> dict:
		## TODO: Cached

		# Ensure Updated Geometry
		log.debug('Updating View Layer')
		bpy.context.view_layer.update()

		# Compute Evaluted + Triangulated Mesh
		log.debug('Casting BMesh of "%s" to Temporary Mesh', self.name)
		_mesh = bpy.data.meshes.new(name='TemporaryMesh')
		with self.mesh_as_bmesh(evaluate=True, triangulate=True) as bmesh_mesh:
			bmesh_mesh.to_mesh(_mesh)

		# Optimized Vertex Copy
		## See <https://blog.michelanders.nl/2016/02/copying-vertices-to-numpy-arrays-in_4.html>
		log.debug('Copying Vertices from "%s"', self.name)
		verts = np.zeros(3 * len(_mesh.vertices), dtype=np.float64)
		_mesh.vertices.foreach_get('co', verts)
		verts.shape = (-1, 3)

		# Optimized Triangle Copy
		## To understand, read it, **carefully**.
		log.debug('Copying Faces from "%s"', self.name)
		faces = np.zeros(3 * len(_mesh.polygons), dtype=np.uint64)
		_mesh.polygons.foreach_get('vertices', faces)
		faces.shape = (-1, 3)

		# Remove Temporary Mesh
		log.debug('Removing Temporary Mesh')
		bpy.data.meshes.remove(_mesh)

		return {
			'verts': verts,
			'faces': faces,
		}

	####################
	# - Modifiers
	####################
	def bl_modifier(
		self,
		modifier_type: ModifierType,
	):
		"""Creates a new modifier for the current `bl_object`.

		- Modifier Type Names: <https://docs.blender.org/api/current/bpy_types_enum_items/object_modifier_type_items.html#rna-enum-object-modifier-type-items>
		"""
		if not (bl_object := bpy.data.objects.get(self.name)):
			msg = f'Tried to add modifier to "{self.name}", but it has no bl_object'
			raise ValueError(msg)

		# (Create and) Return Modifier
		bl_modifier_name = MODIFIER_NAMES[modifier_type]
		if bl_modifier_name not in bl_object.modifiers:
			return bl_object.modifiers.new(
				name=bl_modifier_name,
				type=modifier_type,
			)
		return bl_object.modifiers[bl_modifier_name]

	def modifier_attrs(self, modifier_type: ModifierType) -> dict:
		"""Based on the modifier type, retrieve a representative dictionary of modifier attributes.
		The attributes can then easily be set using `setattr`.
		"""
		bl_modifier = self.bl_modifier(modifier_type)

		if modifier_type == 'NODES':
			return {
				'node_group': bl_modifier.node_group,
			}
		elif modifier_type == 'ARRAY':
			raise NotImplementedError

	def s_modifier_attrs(
		self,
		modifier_type: ModifierType,
		modifier_attrs: dict,
	):
		bl_modifier = self.bl_modifier(modifier_type)

		if modifier_type == 'NODES':
			if bl_modifier.node_group != modifier_attrs['node_group']:
				bl_modifier.node_group = modifier_attrs['node_group']
		elif modifier_type == 'ARRAY':
			raise NotImplementedError

	####################
	# - GeoNodes Modifier
	####################
	def sync_geonodes_modifier(
		self,
		geonodes_node_group,
		geonodes_identifier_to_value: dict,
	):
		"""Push the given GeoNodes Interface values to a GeoNodes modifier attached to a managed MESH object.

		The values must be compatible with the `default_value`s of the interface sockets.

		If there is no object, it is created.
		If the object isn't a MESH object, it is made so.
		If the GeoNodes modifier doesn't exist, it is created.
		If the GeoNodes node group doesn't match, it is changed.
		Only differing interface values are actually changed.
		"""
		bl_object = self.bl_object('MESH')

		# Get (/make) a GeoModes Modifier
		bl_modifier = self.bl_modifier('NODES')

		# Set GeoNodes Modifier Attributes (specifically, the 'node_group')
		self.s_modifier_attrs('NODES', {'node_group': geonodes_node_group})

		# Set GeoNodes Values
		modifier_altered = False
		for (
			interface_identifier,
			value,
		) in geonodes_identifier_to_value.items():
			if bl_modifier[interface_identifier] != value:
				# Quickly Determine if IDPropertyArray is Equal
				if (
					hasattr(bl_modifier[interface_identifier], 'to_list')
					and tuple(bl_modifier[interface_identifier].to_list()) == value
				):
					continue

				# Quickly Determine int/float Mismatch
				if isinstance(
					bl_modifier[interface_identifier],
					float,
				) and isinstance(value, int):
					value = float(value)

				bl_modifier[interface_identifier] = value

				modifier_altered = True

		# Update DepGraph (if anything changed)
		if modifier_altered:
			bl_object.data.update()
