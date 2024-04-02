import contextlib

import bmesh
import bpy
import numpy as np

from ....utils import logger
from .. import contracts as ct
from .managed_bl_collection import managed_collection, preview_collection

log = logger.get(__name__)


####################
# - BLMesh
####################
class ManagedBLMesh(ct.schemas.ManagedObj):
	managed_obj_type = ct.ManagedObjType.ManagedBLMesh
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
			'Changing BLMesh w/Name "%s" to Name "%s"', self._bl_object_name, value
		)

		if not bpy.data.objects.get(value):
			log.info(
				'Desired BLMesh Name "%s" Not Taken',
				value,
			)

			if self._bl_object_name is None:
				log.info(
					'Set New BLMesh Name to "%s"',
					value,
				)
			elif bl_object := bpy.data.objects.get(self._bl_object_name):
				log.info(
					'Changed BLMesh Name to "%s"',
					value,
				)
				bl_object.name = value
			else:
				msg = f'ManagedBLMesh with name "{self._bl_object_name}" was deleted'
				raise RuntimeError(msg)

			# Set Internal Name
			self._bl_object_name = value
		else:
			log.info(
				'Desired BLMesh Name "%s" is Taken. Using Blender Rename',
				value,
			)

			# Set Name Anyway, but Respect Blender's Renaming
			## When a name already exists, Blender adds .### to prevent overlap.
			## `set_name` is allowed to change the name; nodes account for this.
			bl_object.name = value
			self._bl_object_name = bl_object.name

			log.info(
				'Changed BLMesh Name to "%s"',
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
		log.info('Removing "%s" BLMesh', bl_object.type)
		bpy.data.meshes.remove(bl_object.data)

	####################
	# - Actions
	####################
	def show_preview(self) -> None:
		"""Moves the managed Blender object to the preview collection.

		If it's already included, do nothing.
		"""
		if (
			bl_object := bpy.data.objects.get(self.name)
		) is not None and bl_object.name not in preview_collection().objects:
			log.info('Moving "%s" to Preview Collection', bl_object.name)
			preview_collection().objects.link(bl_object)

		msg = 'Managed BLMesh does not exist'
		raise ValueError(msg)

	def hide_preview(self) -> None:
		"""Removes the managed Blender object from the preview collection.

		If it's already removed, do nothing.
		"""
		if (
			bl_object := bpy.data.objects.get(self.name)
		) is not None and bl_object.name in preview_collection().objects:
			log.info('Removing "%s" from Preview Collection', bl_object.name)
			preview_collection.objects.unlink(bl_object)

		msg = 'Managed BLMesh does not exist'
		raise ValueError(msg)

	def bl_select(self) -> None:
		"""Selects the managed Blender object, causing it to be ex. outlined in the 3D viewport."""
		if (bl_object := bpy.data.objects.get(self.name)) is not None:
			bpy.ops.object.select_all(action='DESELECT')
			bl_object.select_set(True)

		msg = 'Managed BLMesh does not exist'
		raise ValueError(msg)

	####################
	# - BLMesh Management
	####################
	def bl_object(self, location: tuple[float, float, float] = (0, 0, 0)):
		"""Returns the managed blender object."""
		# Create Object w/Appropriate Data Block
		if not (bl_object := bpy.data.objects.get(self.name)):
			log.info(
				'Creating BLMesh Object "%s"',
				bl_object.name,
			)
			bl_data = bpy.data.meshes.new(self.name)
			bl_object = bpy.data.objects.new(self.name, bl_data)
			log.debug(
				'Linking "%s" to Base Collection',
				bl_object.name,
			)
			managed_collection().objects.link(bl_object)

		for i, coord in enumerate(location):
			if bl_object.location[i] != coord:
				bl_object.location[i] = coord

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
		if bl_object := bpy.data.objects.get(self.name):
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