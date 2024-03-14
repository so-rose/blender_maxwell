import typing as typ
import typing_extensions as typx
import io

import numpy as np
import pydantic as pyd
import matplotlib.axis as mpl_ax

import bpy

from .. import contracts as ct

AREA_TYPE = "IMAGE_EDITOR"
SPACE_TYPE = "IMAGE_EDITOR"

class ManagedBLImage(ct.schemas.ManagedObj):
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
			else:
				msg = f"Desired name {value} for BL image is taken"
				raise ValueError(msg)
		
		# Object DOES Exist
		bl_image.name = value
		self._bl_image_name = bl_image.name
		## - When name exists, Blender adds .### to prevent overlap.
		## - `set_name` is allowed to change the name; nodes account for this.
	
	def free(self):
		if not (bl_image := bpy.data.images.get(self.name)):
			msg = "Can't free BL image that doesn't exist"
			raise ValueError(msg)
		
		bpy.data.images.remove(bl_image)
	
	####################
	# - Managed Object Management
	####################
	def bl_image(
		self,
		width_px: int,
		height_px: int,
		color_model: typx.Literal["RGB", "RGBA"],
		dtype: typx.Literal["uint8", "float32"],
	):
		"""Returns the managed blender image.
		
		If the requested image properties are different from the image's, then delete the old image make a new image with correct geometry.
		"""
		channels = 4 if color_model == "RGBA" else 3
		
		# Remove Image (if mismatch)
		if (
			(bl_image := bpy.data.images.get(self.name))
			and (
				bl_image.size[0] != width_px
				or bl_image.size[1] != height_px
				or bl_image.channels != channels
				or bl_image.is_float ^ (dtype == "float32")
			)
		):
			self.free()
		
		# Create Image w/Geometry (if none exists)
		if not (bl_image := bpy.data.images.get(self.name)):
			bl_image = bpy.data.images.new(
				self.name,
				width=width_px,
				height=height_px,
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
			area
			for area in bpy.context.screen.areas
			if area.type == AREA_TYPE
		]
		if valid_areas:
			return valid_areas[0]
	
	@property
	def preview_space(self) -> bpy.types.SpaceProperties:
		"""Returns the visible preview space in the visible preview area of
		the Blender UI
		"""
		if (preview_area := self.preview_area):
			return next(
				space
				for space in preview_area.spaces
				if space.type == SPACE_TYPE
			)
	
	####################
	# - Actions
	####################
	def bl_select(self) -> None:
		"""Synchronizes the managed object to the preview, by manipulating
		relevant editors.
		"""
		if (bl_image := bpy.data.images.get(self.name)):
			self.preview_space.image = bl_image
	
	####################
	# - Special Methods
	####################
	def mpl_plot_to_image(
		self,
		func_plotter: typ.Callable[[mpl_ax.Axis], None],
		width_inches: float | None = None,
		height_inches: float | None = None,
		dpi: int | None = None,
		bl_select: bool = False,
	):
		import matplotlib.pyplot as plt
		
		# Compute Image Geometry
		if (preview_area := self.preview_area):
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
			msg = f"There must either be a preview area, or defined `width_inches`, `height_inches`, and `dpi`"
			raise ValueError(msg)
		
		# Compute Plot Dimensions
		aspect_ratio = _width_inches / _height_inches
		
		# Create MPL Figure, Axes, and Compute Figure Geometry
		fig, ax = plt.subplots(
			figsize=[_width_inches, _height_inches],
			dpi=_dpi,
		)
		ax.set_aspect(aspect_ratio)
		cmp_width_px, cmp_height_px = fig.canvas.get_width_height()
		## Use computed pixel w/h to preempt off-by-one size errors.
		
		# Plot w/User Parameter
		func_plotter(ax)
		
		# Save Figure to BytesIO
		with io.BytesIO() as buff:
			fig.savefig(buff, format='raw', dpi=dpi)
			buff.seek(0)
			image_data = np.frombuffer(
				buff.getvalue(),
				dtype=np.uint8,
			).reshape([cmp_height_px, cmp_width_px, -1])
			
			image_data = np.flipud(image_data).astype(np.float32) / 255
		plt.close(fig)
		
		# Optimized Write to Blender Image
		bl_image = self.bl_image(cmp_width_px, cmp_height_px, "RGBA", "uint8")
		bl_image.pixels.foreach_set(image_data.ravel())
		bl_image.update()
		
		if bl_select:
			self.bl_select()
		
