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

import contextlib

import bmesh
import bpy
import numpy as np

from blender_maxwell.utils import logger

from .. import contracts as ct
from . import base
from .managed_bl_collection import managed_collection, preview_collection

log = logger.get(__name__)


####################
# - BLMesh
####################
class ManagedBLMesh(base.ManagedObj):
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
		log.debug(
			'Changing BLMesh w/Name "%s" to Name "%s"', self._bl_object_name, value
		)

		existing_bl_object = bpy.data.objects.get(self.name)

		# No Existing Object: Set Value to Name
		if existing_bl_object is None:
			self._bl_object_name = value

		# Existing Object: Rename to New Name
		else:
			existing_bl_object.name = value
			self._bl_object_name = value

			# Check: Blender Rename -> Synchronization Error
			## -> We can't do much else than report to the user & free().
			if existing_bl_object.name != self._bl_object_name:
				log.critical(
					'BLMesh: Failed to set name of %s to %s, as %s already exists.'
				)
				self._bl_object_name = existing_bl_object.name
				self.free()

	####################
	# - Allocation
	####################
	def __init__(self, name: str, prev_name: str | None = None):
		if prev_name is not None:
			self._bl_object_name = prev_name
		else:
			self._bl_object_name = name

		self.name = name

	####################
	# - Deallocation
	####################
	def free(self):
		bl_object = bpy.data.objects.get(self.name)
		if bl_object is None:
			return

		# Delete the Mesh Datablock
		## -> This automatically deletes the object too
		log.info('BLMesh: Freeing "%s"', self.name)
		bpy.data.meshes.remove(bl_object.data)

	####################
	# - Methods
	####################
	def show_preview(self) -> None:
		"""Moves the managed Blender object to the preview collection.

		If it's already included, do nothing.
		"""
		bl_object = bpy.data.objects.get(self.name)
		if bl_object is None:
			log.info('%s (ManagedBLMesh): Created BLObject for Preview', bl_object.name)
			bl_object = self.bl_object()

		if bl_object.name not in preview_collection().objects:
			log.info('Moving "%s" to Preview Collection', bl_object.name)
			preview_collection().objects.link(bl_object)

	def hide_preview(self) -> None:
		"""Hide any active preview of the managed object, if it exists, and if such an operation makes sense."""
		bl_object = bpy.data.objects.get(self.name)
		if bl_object is not None and bl_object.name in preview_collection().objects:
			log.info('Removing "%s" from Preview Collection', bl_object.name)
			preview_collection().objects.unlink(bl_object)

	def bl_select(self) -> None:
		"""Select the managed object in Blender, if it exists, and if such an operation makes sense."""
		bl_object = bpy.data.objects.get(self.name)
		if bl_object is not None:
			bpy.ops.object.select_all(action='DESELECT')
			bl_object.select_set(True)

	####################
	# - BLMesh Management
	####################
	def bl_object(self, location: tuple[float, float, float] = (0, 0, 0)):
		"""Returns the managed blender object."""
		# Create Object w/Appropriate Data Block
		if not (bl_object := bpy.data.objects.get(self.name)):
			log.info(
				'Creating BLMesh Object "%s"',
				self.name,
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
