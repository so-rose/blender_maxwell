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

"""Declares `ManagedBLText`."""

import time
import typing as typ

import bpy
import matplotlib.axis as mpl_ax
import numpy as np

from blender_maxwell.utils import image_ops, logger

from .. import contracts as ct
from . import base

log = logger.get(__name__)

AREA_TYPE = 'IMAGE_EDITOR'
SPACE_TYPE = 'IMAGE_EDITOR'


####################
# - Managed BL Image
####################
class ManagedBLText(base.ManagedObj):
	"""Represents a Blender Image datablock, encapsulating various useful interactions with it.

	Attributes:
		name: The name of the image.
	"""

	managed_obj_type = ct.ManagedObjType.ManagedBLText
	_bl_image_name: str

	def __init__(self, name: str, prev_name: str | None = None):
		if prev_name is not None:
			self._bl_image_name = prev_name
		else:
			self._bl_image_name = name

		self.name = name

	@property
	def name(self):
		return self._bl_image_name

	@name.setter
	def name(self, value: str):
		log.info(
			'Changing ManagedBLText from "%s" to "%s"',
			self.name,
			value,
		)
		existing_bl_image = bpy.data.images.get(self.name)

		# No Existing Image: Set Value to Name
		if existing_bl_image is None:
			self._bl_image_name = value

		# Existing Image: Rename to New Name
		else:
			existing_bl_image.name = value
			self._bl_image_name = value

			# Check: Blender Rename -> Synchronization Error
			## -> We can't do much else than report to the user & free().
			if existing_bl_image.name != self._bl_image_name:
				log.critical(
					'BLImage: Failed to set name of %s to %s, as %s already exists.'
				)
				self._bl_image_name = existing_bl_image.name
				self.free()

	def free(self):
		bl_image = bpy.data.images.get(self.name)
		if bl_image is not None:
			log.debug('Freeing ManagedBLText "%s"', self.name)
			bpy.data.images.remove(bl_image)

	####################
	# - Managed Object Management
	####################
	def bl_image(
		self,
		width_px: int,
		height_px: int,
		color_model: typ.Literal['RGB', 'RGBA'],
		dtype: typ.Literal['uint8', 'float32'],
	):
		"""Returns the managed blender image.

		If the requested image properties are different from the image's, then delete the old image make a new image with correct geometry.
		"""
		channels = 4 if color_model == 'RGBA' else 3

		# Remove Image (if mismatch)
		bl_image = bpy.data.images.get(self.name)
		if bl_image is not None and (
			bl_image.size[0] != width_px
			or bl_image.size[1] != height_px
			or bl_image.channels != channels
			or bl_image.is_float ^ (dtype == 'float32')
		):
			self.free()

		# Create Image w/Geometry (if none exists)
		bl_image = bpy.data.images.get(self.name)
		if bl_image is None:
			bl_image = bpy.data.images.new(
				self.name,
				width=width_px,
				height=height_px,
				float_buffer=dtype == 'float32',
			)

			# Enable Fake User
			bl_image.use_fake_user = True

		return bl_image

	####################
	# - Editor UX Manipulation
	####################
	@classmethod
	def preview_area(cls) -> bpy.types.Area | None:
		"""Deduces a Blender UI area that can be used for image preview.

		Returns:
			A Blender UI area, if an appropriate one is visible; else `None`,
		"""
		valid_areas = [
			area for area in bpy.context.screen.areas if area.type == AREA_TYPE
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
				space for space in preview_area.spaces if space.type == SPACE_TYPE
			]
			if valid_spaces:
				return valid_spaces[0]
			return None
		return None

	####################
	# - Methods
	####################
	def bl_select(self) -> None:
		"""Selects the image by loading it into an on-screen UI area/space.

		Notes:
			The image must already be available, else nothing will happen.
		"""
		bl_image = bpy.data.images.get(self.name)
		if bl_image is not None:
			self.preview_space().image = bl_image

	@classmethod
	def hide_preview(cls) -> None:
		"""Deselects the image by loading `None` into the on-screen UI area/space."""
		cls.preview_space().image = None
