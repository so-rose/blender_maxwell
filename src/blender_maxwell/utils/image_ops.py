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

"""Useful image processing operations for use in the addon."""

import enum
import functools
import time
import typing as typ

import jax
import jax.numpy as jnp
import jaxtyping as jtyp
import matplotlib
import matplotlib.axis as mpl_ax
import matplotlib.backends.backend_agg
import matplotlib.figure
import matplotlib.style as mplstyle

from blender_maxwell import contracts as ct
from blender_maxwell.utils import logger

mplstyle.use('fast')  ## TODO: Does this do anything?

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


####################
# - MPL Helpers
####################
@functools.lru_cache(maxsize=16)
def mpl_fig_canvas_ax(width_inches: float, height_inches: float, dpi: int):
	fig = matplotlib.figure.Figure(figsize=[width_inches, height_inches], dpi=dpi)
	canvas = matplotlib.backends.backend_agg.FigureCanvasAgg(fig)
	ax = fig.add_subplot()

	# The Customer is Always Right (in Matters of Taste)
	# fig.tight_layout(pad=0)
	return (fig, canvas, ax)


####################
# - Plotters
####################
# () -> ℝ
def plot_hist_1d(
	data: jtyp.Float32[jtyp.Array, ' size'], info, ax: mpl_ax.Axis
) -> None:
	y_name = info.output_name
	y_unit = info.output_unit

	ax.hist(data, bins=30, alpha=0.75)
	ax.set_title('Histogram')
	ax.set_ylabel(f'{y_name}' + (f'({y_unit})' if y_unit is not None else ''))


# (ℤ) -> ℝ
def plot_box_plot_1d(
	data: jtyp.Float32[jtyp.Array, ' heights'], info, ax: mpl_ax.Axis
) -> None:
	x_name = info.dim_names[0]
	y_name = info.output_name
	y_unit = info.output_unit

	ax.boxplot(data)
	ax.set_title('Box Plot')
	ax.set_xlabel(f'{x_name}')
	ax.set_ylabel(f'{y_name}' + (f'({y_unit})' if y_unit is not None else ''))


# (ℝ) -> ℝ
def plot_curve_2d(
	data: jtyp.Float32[jtyp.Array, ' points'], info, ax: mpl_ax.Axis
) -> None:
	times = [time.perf_counter()]

	x_name = info.dim_names[0]
	x_unit = info.dim_units[x_name]
	y_name = info.output_name
	y_unit = info.output_unit

	times.append(time.perf_counter() - times[0])
	ax.plot(info.dim_idx_arrays[0], data)
	times.append(time.perf_counter() - times[0])
	ax.set_title('2D Curve')
	times.append(time.perf_counter() - times[0])
	ax.set_xlabel(f'{x_name}' + (f'({x_unit})' if x_unit is not None else ''))
	times.append(time.perf_counter() - times[0])
	ax.set_ylabel(f'{y_name}' + (f'({y_unit})' if y_unit is not None else ''))
	times.append(time.perf_counter() - times[0])
	# log.critical('Timing of Curve2D: %s', str(times))


def plot_points_2d(
	data: jtyp.Float32[jtyp.Array, ' points'], info, ax: mpl_ax.Axis
) -> None:
	x_name = info.dim_names[0]
	x_unit = info.dim_units[x_name]
	y_name = info.output_name
	y_unit = info.output_unit

	ax.scatter(info.dim_idx_arrays[0], data, alpha=0.6)
	ax.set_title('2D Points')
	ax.set_xlabel(f'{x_name}' + (f'({x_unit})' if x_unit is not None else ''))
	ax.set_ylabel(f'{y_name}' + (f'({y_unit})' if y_unit is not None else ''))


def plot_bar(data: jtyp.Float32[jtyp.Array, ' points'], info, ax: mpl_ax.Axis) -> None:
	x_name = info.dim_names[0]
	x_unit = info.dim_units[x_name]
	y_name = info.output_name
	y_unit = info.output_unit

	ax.bar(info.dim_idx_arrays[0], data, alpha=0.7)
	ax.set_title('2D Bar')
	ax.set_xlabel(f'{x_name}' + (f'({x_unit})' if x_unit is not None else ''))
	ax.set_ylabel(f'{y_name}' + (f'({y_unit})' if y_unit is not None else ''))


# (ℝ, ℤ) -> ℝ
def plot_curves_2d(
	data: jtyp.Float32[jtyp.Array, 'x_size categories'], info, ax: mpl_ax.Axis
) -> None:
	x_name = info.dim_names[0]
	x_unit = info.dim_units[x_name]
	y_name = info.output_name
	y_unit = info.output_unit

	for category in range(data.shape[1]):
		ax.plot(info.dim_idx_arrays[0], data[:, category])

	ax.set_title('2D Curves')
	ax.set_xlabel(f'{x_name}' + (f'({x_unit})' if x_unit is not None else ''))
	ax.set_ylabel(f'{y_name}' + (f'({y_unit})' if y_unit is not None else ''))
	ax.legend()


def plot_filled_curves_2d(
	data: jtyp.Float32[jtyp.Array, 'x_size 2'], info, ax: mpl_ax.Axis
) -> None:
	x_name = info.dim_names[0]
	x_unit = info.dim_units[x_name]
	y_name = info.output_name
	y_unit = info.output_unit

	shared_x_idx = info.dim_idx_arrays[0]
	ax.fill_between(shared_x_idx, data[:, 0], shared_x_idx, data[:, 1])
	ax.set_title('2D Filled Curves')
	ax.set_xlabel(f'{x_name}' + (f'({x_unit})' if x_unit is not None else ''))
	ax.set_ylabel(f'{y_name}' + (f'({y_unit})' if y_unit is not None else ''))


# (ℝ, ℝ) -> ℝ
def plot_heatmap_2d(
	data: jtyp.Float32[jtyp.Array, 'x_size y_size'], info, ax: mpl_ax.Axis
) -> None:
	x_name = info.dim_names[0]
	x_unit = info.dim_units[x_name]
	y_name = info.dim_names[1]
	y_unit = info.dim_units[y_name]

	heatmap = ax.imshow(data, aspect='auto', interpolation='none')
	# ax.figure.colorbar(heatmap, ax=ax)
	ax.set_title('Heatmap')
	ax.set_xlabel(f'{x_name}' + (f'({x_unit})' if x_unit is not None else ''))
	ax.set_ylabel(f'{y_name}' + (f'({y_unit})' if y_unit is not None else ''))
