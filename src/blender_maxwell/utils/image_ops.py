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
import typing as typ

import jax
import jax.numpy as jnp
import jaxtyping as jtyp
import matplotlib
import matplotlib.axis as mpl_ax
import matplotlib.backends.backend_agg
import matplotlib.figure
import matplotlib.style as mplstyle
import seaborn as sns

from blender_maxwell import contracts as ct
from blender_maxwell.utils import logger

# mplstyle.use('fast')  ## TODO: Does this do anything?
sns.set_theme()

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

	####################
	# - UI
	####################
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
	fig = matplotlib.figure.Figure(
		figsize=[width_inches, height_inches], dpi=dpi, layout='tight'
	)
	canvas = matplotlib.backends.backend_agg.FigureCanvasAgg(fig)
	ax = fig.add_subplot()

	# The Customer is Always Right (in Matters of Taste)
	# fig.tight_layout(pad=0)
	return (fig, canvas, ax)


####################
# - Plotters
####################
# (ℤ) -> ℝ
def plot_box_plot_1d(data, ax: mpl_ax.Axis) -> None:
	x_sym, y_sym = list(data.keys())

	ax.boxplot([data[y_sym]])
	ax.set_title(f'{x_sym.name_pretty} -> {y_sym.name_pretty}')
	ax.set_xlabel(x_sym.plot_label)
	ax.set_xlabel(y_sym.plot_label)


def plot_bar(data, ax: mpl_ax.Axis) -> None:
	x_sym, heights_sym = list(data.keys())

	p = ax.bar(data[x_sym], data[heights_sym])
	ax.bar_label(p, label_type='center')

	ax.set_title(f'{x_sym.name_pretty} -> {heights_sym.name_pretty}')
	ax.set_xlabel(x_sym.plot_label)
	ax.set_xlabel(heights_sym.plot_label)


# (ℝ) -> ℝ
def plot_curve_2d(data, ax: mpl_ax.Axis) -> None:
	x_sym, y_sym = list(data.keys())

	ax.plot(data[x_sym], data[y_sym])
	ax.set_title(f'{x_sym.name_pretty} -> {y_sym.name_pretty}')
	ax.set_xlabel(x_sym.plot_label)
	ax.set_xlabel(y_sym.plot_label)


def plot_points_2d(data, ax: mpl_ax.Axis) -> None:
	x_sym, y_sym = list(data.keys())

	ax.scatter(data[x_sym], data[y_sym])
	ax.set_title(f'{x_sym.name_pretty} -> {y_sym.name_pretty}')
	ax.set_xlabel(x_sym.plot_label)
	ax.set_xlabel(y_sym.plot_label)


# (ℝ, ℤ) -> ℝ
def plot_curves_2d(data, ax: mpl_ax.Axis) -> None:
	x_sym, label_sym, y_sym = list(data.keys())

	for i, label in enumerate(data[label_sym]):
		ax.plot(data[x_sym], data[y_sym][:, i], label=label)

	ax.set_title(f'{x_sym.name_pretty} -> {y_sym.name_pretty}')
	ax.set_xlabel(x_sym.plot_label)
	ax.set_xlabel(y_sym.plot_label)
	ax.legend()


def plot_filled_curves_2d(
	data: jtyp.Float32[jtyp.Array, 'x_size 2'], info, ax: mpl_ax.Axis
) -> None:
	x_sym, _, y_sym = list(data.keys())

	ax.fill_between(data[x_sym], data[y_sym][:, 0], data[x_sym], data[y_sym][:, 1])
	ax.set_title(f'{x_sym.name_pretty} -> {y_sym.name_pretty}')
	ax.set_xlabel(x_sym.plot_label)
	ax.set_xlabel(y_sym.plot_label)
	ax.legend()


# (ℝ, ℝ) -> ℝ
def plot_heatmap_2d(
	data: jtyp.Float32[jtyp.Array, 'x_size y_size'], info, ax: mpl_ax.Axis
) -> None:
	x_sym, y_sym, c_sym = list(data.keys())

	heatmap = ax.imshow(data[c_sym], aspect='equal', interpolation='none')
	ax.figure.colorbar(heatmap, cax=ax)

	ax.set_title(f'({x_sym.name_pretty}, {y_sym.name_pretty}) -> {c_sym.plot_label}')
	ax.set_xlabel(x_sym.plot_label)
	ax.set_xlabel(y_sym.plot_label)
	ax.legend()
