import io
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
	managed_obj_type = ct.ManagedObjType.ManagedBLImage
	_bl_image_name: str

	def __init__(self, name: str):
		## TODO: Check that blender doesn't have any other images by the same name.
		self._bl_image_name = name

	@property
	def name(self):
		return self._bl_image_name

	@name.setter
	def name(self, value: str):
		# Image Doesn't Exist
		if not (bl_image := bpy.data.images.get(self._bl_image_name)):
			# ...AND Desired Image Name is Not Taken
			if not bpy.data.objects.get(value):
				self._bl_image_name = value
				return

			# ...AND Desired Image Name is Taken
			msg = f'Desired name {value} for BL image is taken'
			raise ValueError(msg)

		# Object DOES Exist
		bl_image.name = value
		self._bl_image_name = bl_image.name
		## - When name exists, Blender adds .### to prevent overlap.
		## - `set_name` is allowed to change the name; nodes account for this.

	def free(self):
		if bl_image := bpy.data.images.get(self.name):
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
		if (bl_image := bpy.data.images.get(self.name)) and (
			bl_image.size[0] != width_px
			or bl_image.size[1] != height_px
			or bl_image.channels != channels
			or bl_image.is_float ^ (dtype == 'float32')
		):
			self.free()

		# Create Image w/Geometry (if none exists)
		if not (bl_image := bpy.data.images.get(self.name)):
			bl_image = bpy.data.images.new(
				self.name,
				width=width_px,
				height=height_px,
				float_buffer=dtype == 'float32',
			)

		return bl_image

	####################
	# - Editor UX Manipulation
	####################
	@property
	def preview_area(self) -> bpy.types.Area:
		"""Returns the visible preview area in the Blender UI.
		If none are valid, return None.
		"""
		valid_areas = [
			area for area in bpy.context.screen.areas if area.type == AREA_TYPE
		]
		if valid_areas:
			return valid_areas[0]

	@property
	def preview_space(self) -> bpy.types.SpaceProperties:
		"""Returns the visible preview space in the visible preview area of
		the Blender UI
		"""
		if preview_area := self.preview_area:
			return next(
				space for space in preview_area.spaces if space.type == SPACE_TYPE
			)

	####################
	# - Methods
	####################
	def bl_select(self) -> None:
		"""Synchronizes the managed object to the preview, by manipulating
		relevant editors.
		"""
		if bl_image := bpy.data.images.get(self.name):
			self.preview_space.image = bl_image

	def hide_preview(self) -> None:
		self.preview_space.image = None

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
		if preview_area := self.preview_area:
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
		# log.debug('Computed Image Data (%f)', time.perf_counter() - time_start)

		bl_image = self.bl_image(width_px, height_px, 'RGBA', 'float32')
		bl_image.pixels.foreach_set(np.float32(image_data).ravel())
		bl_image.update()
		# log.debug('Set BL Image (%f)', time.perf_counter() - time_start)

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
		# time_start = time.perf_counter()
		import matplotlib.pyplot as plt
		# log.debug('Imported PyPlot (%f)', time.perf_counter() - time_start)

		# Compute Plot Dimensions
		aspect_ratio, _dpi, _width_inches, _height_inches, width_px, height_px = (
			self.gen_image_geometry(width_inches, height_inches, dpi)
		)
		# log.debug('Computed MPL Geometry (%f)', time.perf_counter() - time_start)

		# log.debug(
		# 'Creating MPL Axes (aspect=%f, width=%f, height=%f)',
		# aspect_ratio,
		# _width_inches,
		# _height_inches,
		# )
		# Create MPL Figure, Axes, and Compute Figure Geometry
		fig, ax = plt.subplots(
			figsize=[_width_inches, _height_inches],
			dpi=_dpi,
		)
		# log.debug('Created MPL Axes (%f)', time.perf_counter() - time_start)
		ax.set_aspect(aspect_ratio)
		cmp_width_px, cmp_height_px = fig.canvas.get_width_height()
		## Use computed pixel w/h to preempt off-by-one size errors.
		ax.set_aspect('auto')  ## Workaround aspect-ratio bugs
		# log.debug('Set MPL Aspect (%f)', time.perf_counter() - time_start)

		# Plot w/User Parameter
		func_plotter(ax)
		# log.debug('User Plot Function (%f)', time.perf_counter() - time_start)

		# Save Figure to BytesIO
		with io.BytesIO() as buff:
			# log.debug('Made BytesIO (%f)', time.perf_counter() - time_start)
			fig.savefig(buff, format='raw', dpi=dpi)
			# log.debug('Saved Figure to BytesIO (%f)', time.perf_counter() - time_start)
			buff.seek(0)
			image_data = np.frombuffer(
				buff.getvalue(),
				dtype=np.uint8,
			).reshape([cmp_height_px, cmp_width_px, -1])
			# log.debug('Set Image Data (%f)', time.perf_counter() - time_start)

			image_data = np.flipud(image_data).astype(np.float32) / 255
			# log.debug('Flipped Image Data (%f)', time.perf_counter() - time_start)
		plt.close(fig)

		# Optimized Write to Blender Image
		bl_image = self.bl_image(cmp_width_px, cmp_height_px, 'RGBA', 'uint8')
		# log.debug('Made BL Image (%f)', time.perf_counter() - time_start)
		bl_image.pixels.foreach_set(image_data.ravel())
		# log.debug('Set BL Image Pixels (%f)', time.perf_counter() - time_start)
		bl_image.update()
		# log.debug('Updated BL Image (%f)', time.perf_counter() - time_start)

		if bl_select:
			self.bl_select()
