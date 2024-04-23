"""Useful image processing operations for use in the addon."""

import enum
import typing as typ

import jax
import jax.numpy as jnp
import jaxtyping as jtyp
import matplotlib

from blender_maxwell import contracts as ct
from blender_maxwell.utils import logger

log = logger.get(__name__)

####################
# - Constants
####################
_MPL_CM = matplotlib.cm.get_cmap('viridis', 512)
VIRIDIS_COLORMAP: jtyp.Float32[jtyp.Array, '512 3'] = jnp.array(
	[_MPL_CM(i)[:3] for i in range(512)]
)


class Colormap(enum.StrEnum):
	"""Available colormaps.

	Attributes:
		Viridis: Good general-purpose colormap.
		Grayscale: Simple black and white mapping.
	"""

	Viridis = enum.auto()
	Grayscale = enum.auto()

	@staticmethod
	def to_name(value: typ.Self) -> str:
		return {
			Colormap.Viridis: 'Viridis',
			Colormap.Grayscale: 'Grayscale',
		}[value]

	@staticmethod
	def to_icon(value: typ.Self) -> ct.BLIcon:
		return ''


####################
# - Colormap: (X,Y,1 -> Value) -> (X,Y,4 -> Value)
####################
def apply_colormap(
	normalized_data: jtyp.Float32[jtyp.Array, 'width height 4'],
	colormap: jtyp.Float32[jtyp.Array, '512 3'],
):
	# Linear interpolation between colormap points
	n_colors = colormap.shape[0]
	indices = normalized_data * (n_colors - 1)
	lower_idx = jnp.floor(indices).astype(jnp.int32)
	upper_idx = jnp.ceil(indices).astype(jnp.int32)
	alpha = indices - lower_idx

	lower_colors = jax.vmap(lambda i: colormap[i])(lower_idx)
	upper_colors = jax.vmap(lambda i: colormap[i])(upper_idx)

	return (1 - alpha)[..., None] * lower_colors + alpha[..., None] * upper_colors


@jax.jit
def rgba_image_from_2d_map__viridis(map_2d: jtyp.Float32[jtyp.Array, 'width height 4']):
	amplitude = jnp.abs(map_2d)
	amplitude_normalized = (amplitude - amplitude.min()) / (
		amplitude.max() - amplitude.min()
	)
	rgb_array = apply_colormap(amplitude_normalized, VIRIDIS_COLORMAP)
	alpha_channel = jnp.ones_like(amplitude_normalized)
	return jnp.dstack((rgb_array, alpha_channel))


@jax.jit
def rgba_image_from_2d_map__grayscale(
	map_2d: jtyp.Float32[jtyp.Array, 'width height 4'],
):
	amplitude = jnp.abs(map_2d)
	amplitude_normalized = (amplitude - amplitude.min()) / (
		amplitude.max() - amplitude.min()
	)
	rgb_array = jnp.stack([amplitude_normalized] * 3, axis=-1)
	alpha_channel = jnp.ones_like(amplitude_normalized)
	return jnp.dstack((rgb_array, alpha_channel))


def rgba_image_from_2d_map(
	map_2d: jtyp.Float32[jtyp.Array, 'width height 4'], colormap: str | None = None
):
	"""RGBA Image from a map of 2D coordinates to values.

	Parameters:
		map_2d: The 2D value map.

	Returns:
		Image as a JAX array of shape (height, width, 4)
	"""
	if colormap == Colormap.Viridis:
		return rgba_image_from_2d_map__viridis(map_2d)
	if colormap == Colormap.Grayscale:
		return rgba_image_from_2d_map__grayscale(map_2d)

	return rgba_image_from_2d_map__grayscale(map_2d)
