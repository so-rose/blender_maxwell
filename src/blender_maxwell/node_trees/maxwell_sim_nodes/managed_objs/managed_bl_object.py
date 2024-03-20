import typing as typ
import typing_extensions as typx
import functools
import contextlib
import io

import numpy as np
import pydantic as pyd
import matplotlib.axis as mpl_ax

import bpy
import bmesh

from .. import contracts as ct

ModifierType = typx.Literal['NODES', 'ARRAY']
MODIFIER_NAMES = {
	'NODES': 'BLMaxwell_GeoNodes',
	'ARRAY': 'BLMaxwell_Array',
}
MANAGED_COLLECTION_NAME = 'BLMaxwell'
PREVIEW_COLLECTION_NAME = 'BLMaxwell Visible'


def bl_collection(
	collection_name: str, view_layer_exclude: bool
) -> bpy.types.Collection:
	# Init the "Managed Collection"
	# Ensure Collection exists (and is in the Scene collection)
	if collection_name not in bpy.data.collections:
		collection = bpy.data.collections.new(collection_name)
		bpy.context.scene.collection.children.link(collection)
	else:
		collection = bpy.data.collections[collection_name]

	## Ensure synced View Layer exclusion
	if (
		layer_collection := bpy.context.view_layer.layer_collection.children[
			collection_name
		]
	).exclude != view_layer_exclude:
		layer_collection.exclude = view_layer_exclude

	return collection


class ManagedBLObject(ct.schemas.ManagedObj):
	managed_obj_type = ct.ManagedObjType.ManagedBLObject
	_bl_object_name: str

	def __init__(self, name: str):
		self._bl_object_name = name

	# Object Name
	@property
	def name(self):
		return self._bl_object_name

	@name.setter
	def set_name(self, value: str) -> None:
		# Object Doesn't Exist
		if not (bl_object := bpy.data.objects.get(self._bl_object_name)):
			# ...AND Desired Object Name is Not Taken
			if not bpy.data.objects.get(value):
				self._bl_object_name = value
				return

			# ...AND Desired Object Name is Taken
			else:
				msg = f'Desired name {value} for BL object is taken'
				raise ValueError(msg)

		# Object DOES Exist
		bl_object.name = value
		self._bl_object_name = bl_object.name
		## - When name exists, Blender adds .### to prevent overlap.
		## - `set_name` is allowed to change the name; nodes account for this.

	# Object Datablock Name
	@property
	def bl_mesh_name(self):
		return self.name

	@property
	def bl_volume_name(self):
		return self.name

	# Deallocation
	def free(self):
		if not (bl_object := bpy.data.objects.get(self.name)):
			return  ## Nothing to do

		# Delete the Underlying Datablock
		## This automatically deletes the object too
		if bl_object.type == 'MESH':
			bpy.data.meshes.remove(bl_object.data)
		elif bl_object.type == 'EMPTY':
			bpy.data.meshes.remove(bl_object.data)
		elif bl_object.type == 'VOLUME':
			bpy.data.volumes.remove(bl_object.data)
		else:
			msg = f'Type of to-delete `bl_object`, {bl_object.type}, is not valid'
			raise ValueError(msg)

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
		if (
			bl_object.name
			not in (
				preview_collection := bl_collection(
					PREVIEW_COLLECTION_NAME, view_layer_exclude=False
				)
			).objects
		):
			preview_collection.objects.link(bl_object)

		if kind == 'EMPTY' and empty_display_type is not None:
			bl_object.empty_display_type = empty_display_type

	def hide_preview(
		self,
		kind: typx.Literal['MESH', 'EMPTY', 'VOLUME'],
	) -> None:
		"""Removes the managed Blender object from the preview collection.

		If it's already removed, do nothing.
		"""
		bl_object = self.bl_object(kind)
		if (
			bl_object.name
			not in (
				preview_collection := bl_collection(
					PREVIEW_COLLECTION_NAME, view_layer_exclude=False
				)
			).objects
		):
			preview_collection.objects.unlink(bl_object)

	def bl_select(self) -> None:
		"""Selects the managed Blender object globally, causing it to be ex.
		outlined in the 3D viewport.
		"""
		if not (bl_object := bpy.data.objects.get(self.name)):
			msg = 'Managed BLObject does not exist'
			raise ValueError(msg)

		bpy.ops.object.select_all(action='DESELECT')
		bl_object.select_set(True)

	####################
	# - Managed Object Management
	####################
	def bl_object(
		self,
		kind: typx.Literal['MESH', 'EMPTY', 'VOLUME'],
	):
		"""Returns the managed blender object.

		If the requested object data type is different, then delete the old
		object and recreate.
		"""
		# Remove Object (if mismatch)
		if (
			bl_object := bpy.data.objects.get(self.name)
		) and bl_object.type != kind:
			self.free()

		# Create Object w/Appropriate Data Block
		if not (bl_object := bpy.data.objects.get(self.name)):
			if kind == 'MESH':
				bl_data = bpy.data.meshes.new(self.bl_mesh_name)
			elif kind == 'EMPTY':
				bl_data = None
			elif kind == 'VOLUME':
				raise NotImplementedError
			else:
				msg = (
					f'Requested `bl_object` type {bl_object.type} is not valid'
				)
				raise ValueError(msg)

			bl_object = bpy.data.objects.new(self.name, bl_data)
			bl_collection(
				MANAGED_COLLECTION_NAME, view_layer_exclude=True
			).objects.link(bl_object)

		return bl_object

	####################
	# - Mesh Data Properties
	####################
	@property
	def raw_mesh(self) -> bpy.types.Mesh:
		"""Returns the object's raw mesh data.

		Raises an error if the object has no mesh data.
		"""
		if (
			bl_object := bpy.data.objects.get(self.name)
		) and bl_object.type == 'MESH':
			return bl_object.data

		msg = f'Requested MESH data from `bl_object` of type {bl_object.type}'
		raise ValueError(msg)

	@contextlib.contextmanager
	def mesh_as_bmesh(
		self,
		evaluate: bool = True,
		triangulate: bool = False,
	) -> bpy.types.Mesh:
		if (
			bl_object := bpy.data.objects.get(self.name)
		) and bl_object.type == 'MESH':
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
			msg = f'Requested BMesh from `bl_object` of type {bl_object.type}'
			raise ValueError(msg)

	@property
	def mesh_as_arrays(self) -> dict:
		## TODO: Cached

		# Ensure Updated Geometry
		bpy.context.view_layer.update()
		## TODO: Must we?

		# Compute Evaluted + Triangulated Mesh
		_mesh = bpy.data.meshes.new(name='TemporaryMesh')
		with self.mesh_as_bmesh(evaluate=True, triangulate=True) as bmesh_mesh:
			bmesh_mesh.to_mesh(_mesh)

		# Optimized Vertex Copy
		## See <https://blog.michelanders.nl/2016/02/copying-vertices-to-numpy-arrays-in_4.html>
		verts = np.zeros(3 * len(_mesh.vertices), dtype=np.float64)
		_mesh.vertices.foreach_get('co', verts)
		verts.shape = (-1, 3)

		# Optimized Triangle Copy
		## To understand, read it, **carefully**.
		faces = np.zeros(3 * len(_mesh.polygons), dtype=np.uint64)
		_mesh.polygons.foreach_get('vertices', faces)
		faces.shape = (-1, 3)

		# Remove Temporary Mesh
		bpy.data.meshes.remove(_mesh)

		return {
			'verts': verts,
			'faces': faces,
		}

	####################
	# - Modifier Methods
	####################
	def bl_modifier(
		self,
		modifier_type: ModifierType,
	):
		"""Creates a new modifier for the current `bl_object`.

		For all Blender modifier type names, see: <https://docs.blender.org/api/current/bpy_types_enum_items/object_modifier_type_items.html#rna-enum-object-modifier-type-items>
		"""
		if not (bl_object := bpy.data.objects.get(self.name)):
			msg = "Can't add modifier to BL object that doesn't exist"
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
					and tuple(bl_modifier[interface_identifier].to_list())
					== value
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

	# @property
	# def volume(self) -> bpy.types.Volume:
	# """Returns the object's volume data.
	#
	# Raises an error if the object has no volume data.
	# """
	# if (
	# (bl_object := bpy.data.objects.get(self.bl_object_name))
	# and bl_object.type == "VOLUME"
	# ):
	# return bl_object.data
	#
	# msg = f"Requested VOLUME data from `bl_object` of type {bl_object.type}"
	# raise ValueError(msg)
