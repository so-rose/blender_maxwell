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

"""Declares `ManagedBLImage`."""

# import time
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
class ManagedBLImage(base.ManagedObj):
	"""Represents a Blender Image datablock, encapsulating various useful interactions with it.

	Attributes:
		name: The name of the image.
	"""

	managed_obj_type = ct.ManagedObjType.ManagedBLImage
	_bl_image_name: str

	def __init__(self, name: str):
		self._bl_image_name = name

	@property
	def name(self):
		return self._bl_image_name

	@name.setter
	def name(self, value: str):
		log.info(
			'Setting ManagedBLImage from "%s" to "%s"',
			self.name,
			value,
		)
		current_bl_image = bpy.data.images.get(self._bl_image_name)
		wanted_bl_image = bpy.data.images.get(value)

		# Yoink Image Name
		if current_bl_image is None and wanted_bl_image is None:
			self._bl_image_name = value

		# Alter Image Name
		elif current_bl_image is not None and wanted_bl_image is None:
			self._bl_image_name = value
			current_bl_image.name = value

		# Overlapping Image Name
		elif wanted_bl_image is not None:
			msg = f'ManagedBLImage "{self._bl_image_name}" could not change its name to "{value}", since it already exists.'
			raise ValueError(msg)

	def free(self):
		bl_image = bpy.data.images.get(self.name)
		if bl_image is not None:
			log.debug('Freeing ManagedBLImage "%s"', self.name)
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

	####################
	# - Image Geometry
	####################
	def gen_image_geometry(
		self,
		width_inches: float | None = None,
		height_inches: float | None = None,
		dpi: int | None = None,
	):
		# Compute Image Geometry
		if preview_area := self.preview_area():
			# Retrieve DPI from Blender Preferences
			_dpi = bpy.context.preferences.system.dpi

			# Retrieve Image Geometry from Area
			width_px = preview_area.width
			height_px = preview_area.height

			# Compute Inches
			_width_inches = width_px / _dpi
			_height_inches = height_px / _dpi

		elif width_inches and height_inches and dpi:
			# Copy Parameters
			_dpi = dpi
			_width_inches = height_inches
			_height_inches = height_inches

			# Compute Pixel Geometry
			width_px = int(_width_inches * _dpi)
			height_px = int(_height_inches * _dpi)

		else:
			msg = 'There must either be a preview area, or defined `width_inches`, `height_inches`, and `dpi`'
			raise ValueError(msg)

		aspect_ratio = _width_inches / _height_inches

		return aspect_ratio, _dpi, _width_inches, _height_inches, width_px, height_px

	####################
	# - Special Methods
	####################
	def map_2d_to_image(
		self, map_2d, colormap: str | None = 'VIRIDIS', bl_select: bool = False
	):
		self.data_to_image(
			lambda _: image_ops.rgba_image_from_2d_map(map_2d, colormap=colormap),
			bl_select=bl_select,
		)

	def data_to_image(
		self,
		func_image_data: typ.Callable[[int], np.array],
		bl_select: bool = False,
	):
		# time_start = time.perf_counter()
		image_data = func_image_data(4)
		width_px = image_data.shape[1]
		height_px = image_data.shape[0]

		bl_image = self.bl_image(width_px, height_px, 'RGBA', 'float32')
		bl_image.pixels.foreach_set(np.float32(image_data).ravel())
		bl_image.update()

		if bl_select:
			self.bl_select()

	def mpl_plot_to_image(
		self,
		func_plotter: typ.Callable[[mpl_ax.Axis], None],
		width_inches: float | None = None,
		height_inches: float | None = None,
		dpi: int | None = None,
		bl_select: bool = False,
	):
		# times = [time.perf_counter()]

		# Compute Plot Dimensions
		aspect_ratio, _dpi, _width_inches, _height_inches, width_px, height_px = (
			self.gen_image_geometry(width_inches, height_inches, dpi)
		)
		# times.append(['Image Geometry', time.perf_counter() - times[0]])

		# Create MPL Figure, Axes, and Compute Figure Geometry
		fig, canvas, ax = image_ops.mpl_fig_canvas_ax(
			_width_inches, _height_inches, _dpi
		)
		# times.append(['MPL Fig Canvas Axis', time.perf_counter() - times[0]])

		ax.clear()
		# times.append(['Clear Axis', time.perf_counter() - times[0]])

		# Plot w/User Parameter
		func_plotter(ax)
		# times.append(['Plot!', time.perf_counter() - times[0]])

		# Save Figure to BytesIO
		canvas.draw()
		# times.append(['Draw Pixels', time.perf_counter() - times[0]])

		canvas_width_px, canvas_height_px = fig.canvas.get_width_height()
		# times.append(['Get Canvas Dims', time.perf_counter() - times[0]])
		image_data = (
			np.float32(
				np.flipud(
					np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(
						fig.canvas.get_width_height()[::-1] + (4,)
					)
				)
			)
			/ 255
		)
		# times.append(['Load Data from Canvas', time.perf_counter() - times[0]])

		# Optimized Write to Blender Image
		bl_image = self.bl_image(canvas_width_px, canvas_height_px, 'RGBA', 'uint8')
		# times.append(['Get BLImage', time.perf_counter() - times[0]])
		bl_image.pixels.foreach_set(image_data.ravel())
		# times.append(['Set Pixels', time.perf_counter() - times[0]])
		bl_image.update()
		# times.append(['Update BLImage', time.perf_counter() - times[0]])

		if bl_select:
			self.bl_select()
		# times.append(['Select BLImage', time.perf_counter() - times[0]])

		# log.critical('Timing of MPL Plot')
		# for timing in times:
		# log.critical(timing)


@bpy.app.handlers.persistent
def pack_managed_images(_):
	for image in bpy.data.images:
		if image.is_dirty:
			image.pack()
			## TODO: Only pack images declared by a ManagedBLImage


bpy.app.handlers.save_pre.append(pack_managed_images)
