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

class ManagedBLObject(ct.schemas.ManagedObj):
	managed_obj_type = ct.ManagedObjType.ManagedBLObject
	_bl_object_name: str
	
	def __init__(self, name: str):
		## TODO: Check that blender doesn't have any other objects by the same name.
		self._bl_object_name = name
	
	# Object Name
	@property
	def bl_object_name(self):
		return self._bl_object_name
	
	@bl_object_name.setter
	def set_bl_object_name(self, value: str):
		## TODO: Check that blender doesn't have any other objects by the same name.
		if (bl_object := bpy.data.objects.get(self.bl_object_name)):
			bl_object.name = value
		
		self._bl_object_name = value
	
	# Object Datablock Name
	@property
	def bl_mesh_name(self):
		return self.bl_object_name + "Mesh"
	
	@property
	def bl_volume_name(self):
		return self.bl_object_name + "Volume"
	
	# Deallocation
	def free(self):
		if (bl_object := bpy.data.objects.get(self.bl_object_name)):
			# Delete the Underlying Datablock
			if bl_object.type == "MESH":
				bpy.data.meshes.remove(bl_object.data)
			elif bl_object.type == "VOLUME":
				bpy.data.volumes.remove(bl_object.data)
			else:
				msg = f"Type of to-delete `bl_object`, {bl_object.type}, is not valid"
				raise ValueError(msg)
	
	####################
	# - Actions
	####################
	def trigger_action(
		self, 
		action: typx.Literal["report", "enable_previews"],
	):
		if action == "report":
			pass  ## TODO: Cache invalidation.
		
		if action == "enable_previews":

			pass  ## Image "previews" don't need enabling.

	def bl_select(self) -> None:
		"""Selects the managed Blender object globally, causing it to be ex.
		outlined in the 3D viewport.
		"""
		bpy.ops.object.select_all(action='DESELECT')
		bpy.data.objects['Suzanne'].select_set(True)
	
	####################
	# - Managed Object Management
	####################
	def bl_object(
		self,
		kind: typx.Literal["MESH", "VOLUME"],
	):
		"""Returns the managed blender object.
		
		If the requested object data type is different, then delete the old
		object and recreate.
		"""
		# Remove Object (if mismatch)
		if (
			(bl_object := bpy.data.images.get(self.bl_object_name))
			and bl_object.type != kind
		):
			self.free()
		
		# Create Object w/Appropriate Data Block
		if not (bl_object := bpy.data.images.get(self.bl_object_name)):
			if bl_object.type == "MESH":
				bl_data = bpy.data.meshes.new(self.bl_mesh_name)
			elif bl_object.type == "VOLUME":
				raise NotImplementedError
			else:
				msg = f"Requested `bl_object` type {bl_object.type} is not valid"
				raise ValueError(msg)
			
			bl_object = bpy.data.objects.new(self.bl_object_name, bl_data)
		
		return bl_object
	
	####################
	# - Data Properties
	####################
	@property
	def raw_mesh(self) -> bpy.types.Mesh:
		"""Returns the object's raw mesh data.
		
		Raises an error if the object has no mesh data.
		"""
		if (
			(bl_object := bpy.data.objects.get(self.bl_object_name))
			and bl_object.type == "MESH"
		):
			return bl_object.data
	
		msg = f"Requested MESH data from `bl_object` of type {bl_object.type}"
		raise ValueError(msg)
	
	@contextlib.contextmanager
	def as_bmesh(
		self,
		evaluate: bool = True,
		triangulate: bool = False,
	) -> bpy.types.Mesh:
		if (
			(bl_object := bpy.data.objects.get(self.bl_object_name))
			and bl_object.type == "MESH"
		):
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
				if bmesh_mesh: bmesh_mesh.free()
	
		msg = f"Requested BMesh from `bl_object` of type {bl_object.type}"
		raise ValueError(msg)
	
	@functools.cached_property
	def as_arrays(self) -> dict:
		# Compute Evaluted + Triangulated Mesh
		_mesh = bpy.data.meshes.new(name="TemporaryMesh")
		with self.as_bmesh(evaluate=True, triangulate=True) as bmesh_mesh:
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
			"verts": verts,
			"faces": faces,
		}
	
	#@property
	#def volume(self) -> bpy.types.Volume:
	#	"""Returns the object's volume data.
	#	
	#	Raises an error if the object has no volume data.
	#	"""
	#	if (
	#		(bl_object := bpy.data.objects.get(self.bl_object_name))
	#		and bl_object.type == "VOLUME"
	#	):
	#		return bl_object.data
	#
	#	msg = f"Requested VOLUME data from `bl_object` of type {bl_object.type}"
	#	raise ValueError(msg)
