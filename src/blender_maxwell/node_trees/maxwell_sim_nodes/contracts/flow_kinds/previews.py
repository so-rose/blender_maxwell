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

import typing as typ

import bpy
import pydantic as pyd

from blender_maxwell.utils import logger

IMAGE_AREA_TYPE = 'IMAGE_EDITOR'
IMAGE_SPACE_TYPE = 'IMAGE_EDITOR'

log = logger.get(__name__)

####################
# - Global Collection Handling
####################
MANAGED_COLLECTION_NAME = 'BLMaxwell'
PREVIEW_COLLECTION_NAME = 'BLMaxwell Visible'


def collection(collection_name: str, view_layer_exclude: bool) -> bpy.types.Collection:
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


def managed_collection() -> bpy.types.Collection:
	return collection(MANAGED_COLLECTION_NAME, view_layer_exclude=True)


def preview_collection() -> bpy.types.Collection:
	return collection(PREVIEW_COLLECTION_NAME, view_layer_exclude=False)


####################
# - Global Collection Handling
####################
class PreviewsFlow(pyd.BaseModel):
	"""Represents global references to previewable entries."""

	model_config = pyd.ConfigDict(frozen=True)

	bl_image_name: str | None = None
	bl_object_names: frozenset[str] = frozenset()

	####################
	# - Operations
	####################
	def __or__(self, other: typ.Self) -> typ.Self:
		return PreviewsFlow(
			bl_image_name=other.bl_image_name,
			bl_object_names=self.bl_object_names | other.bl_object_names,
		)

	####################
	# - Image Editor UX
	####################
	@classmethod
	def preview_area(cls) -> bpy.types.Area | None:
		"""Deduces a Blender UI area that can be used for image preview.

		Returns:
			A Blender UI area, if an appropriate one is visible; else `None`,
		"""
		valid_areas = [
			area for area in bpy.context.screen.areas if area.type == IMAGE_AREA_TYPE
		]
		if valid_areas:
			return valid_areas[0]
		return None

	@classmethod
	def preview_space(cls) -> bpy.types.SpaceProperties | None:
		"""Deduces a Blender UI space, within `self.preview_area`, that can be used for image preview.

		Returns:
			A Blender UI space within `self.preview_area`, if it isn't None; else, `None`.
		"""
		preview_area = cls.preview_area()
		if preview_area is not None:
			valid_spaces = [
				space for space in preview_area.spaces if space.type == IMAGE_SPACE_TYPE
			]
			if valid_spaces:
				return valid_spaces[0]
			return None
		return None

	####################
	# - Preview Plot
	####################
	@classmethod
	def hide_image_preview(cls) -> None:
		"""Show all image previews in the first available image editor.

		If no image editors are visible, then nothing happens.
		"""
		preview_space = cls.preview_space()
		if preview_space is not None and preview_space.image is not None:
			cls.preview_space().image = None

	def update_image_preview(self) -> None:
		"""Show the image preview in the first available image editor.

		If no image editors are visible, then nothing happens.
		"""
		preview_space = self.preview_space()
		if self.bl_image_name is not None:
			bl_image = bpy.data.images.get(self.bl_image_name)

			# Replace Preview Space Image
			if (
				bl_image is not None
				and preview_space is not None
				and preview_space.image is not bl_image
			):
				preview_space.image = bl_image

			# Remove Image
			if bl_image is None and preview_space.image is not None:
				preview_space.image = None
		elif preview_space.image is not None:
			preview_space.image = None

	####################
	# - Preview Objects
	####################
	@classmethod
	def hide_bl_object_previews(cls) -> None:
		"""Hide all previewed Blender objects."""
		for bl_object_name in [obj.name for obj in preview_collection().objects]:
			bl_object = bpy.data.objects.get(bl_object_name)
			if bl_object is not None and bl_object.name in preview_collection().objects:
				preview_collection().objects.unlink(bl_object)

	def update_bl_object_previews(self) -> frozenset[str]:
		"""Preview objects that need previewing and unpreview objects that need unpreviewing.

		Designed to utilize the least possible amount of operations to achieve a desired set of previewed objects, from the current set of previewed objects.
		"""
		# Deduce Change in Previewed Objects
		## -> Examine the preview collection for all currently previewed names.
		## -> Preview names that shouldn't exist should be added.
		## -> Preview names that should exist should be removed.
		previewed_bl_objects = {obj.name for obj in preview_collection().objects}
		bl_objects_to_remove = previewed_bl_objects - self.bl_object_names
		bl_objects_to_add = self.bl_object_names - previewed_bl_objects

		# Remove Previews
		## -> Never presume that bl_object is already defined.
		for bl_object_name in bl_objects_to_remove:
			bl_object = bpy.data.objects.get(bl_object_name)
			if bl_object is not None and bl_object.name in preview_collection().objects:
				preview_collection().objects.unlink(bl_object)

		# Add Previews
		## -> Never presume that bl_object is already defined.
		for bl_object_name in bl_objects_to_add:
			bl_object = bpy.data.objects.get(bl_object_name)
			if (
				bl_object is not None
				and bl_object.name not in preview_collection().objects
			):
				preview_collection().objects.link(bl_object)
