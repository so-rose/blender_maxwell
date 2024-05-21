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

import enum
import typing as typ

import bpy
import jaxtyping as jtyp
import matplotlib.axis as mpl_ax
import sympy as sp

from blender_maxwell.utils import bl_cache, image_ops, logger, sim_symbols
from blender_maxwell.utils import extra_sympy_units as spux

from ... import contracts as ct
from ... import managed_objs, sockets
from .. import base, events

log = logger.get(__name__)


class VizMode(enum.StrEnum):
	"""Available visualization modes.

	**NOTE**: >1D output dimensions currently have no viz.

	Plots for `() -> ℝ`:
	- Hist1D: Bin-summed distribution.
	- BoxPlot1D: Box-plot describing the distribution.

	Plots for `(ℤ) -> ℝ`:
	- BoxPlots1D: Side-by-side boxplots w/equal y axis.

	Plots for `(ℝ) -> ℝ`:
	- Curve2D: Standard line-curve w/smooth interpolation
	- Points2D: Scatterplot of individual points.
	- Bar: Value to height of a barplot.

	Plots for `(ℝ, ℤ) -> ℝ`:
	- Curves2D: Layered Curve2Ds with unique colors.
	- FilledCurves2D: Layered Curve2Ds with filled space between.

	Plots for `(ℝ, ℝ) -> ℝ`:
	- Heatmap2D: Colormapped image with value at each pixel.

	Plots for `(ℝ, ℝ, ℝ) -> ℝ`:
	- SqueezedHeatmap2D: 3D-embeddable heatmap for when one of the axes is 1.
	- Heatmap3D: Colormapped field with value at each voxel.
	"""

	Hist1D = enum.auto()
	BoxPlot1D = enum.auto()

	Curve2D = enum.auto()
	Points2D = enum.auto()
	Bar = enum.auto()

	Curves2D = enum.auto()
	FilledCurves2D = enum.auto()

	Heatmap2D = enum.auto()

	SqueezedHeatmap2D = enum.auto()
	Heatmap3D = enum.auto()

	@staticmethod
	def valid_modes_for(info: ct.InfoFlow) -> list[typ.Self] | None:
		EMPTY = ()
		Z = spux.MathType.Integer
		R = spux.MathType.Real
		VM = VizMode

		valid_viz_modes = {
			(EMPTY, (None, R)): [VM.Hist1D, VM.BoxPlot1D],
			((Z), (None, R)): [
				VM.Hist1D,
				VM.BoxPlot1D,
			],
			((R,), (None, R)): [
				VM.Curve2D,
				VM.Points2D,
				VM.Bar,
			],
			((R, Z), (None, R)): [
				VM.Curves2D,
				VM.FilledCurves2D,
			],
			((R, R), (None, R)): [
				VM.Heatmap2D,
			],
			((R, R, R), (None, R)): [VM.SqueezedHeatmap2D, VM.Heatmap3D],
		}.get(
			(
				tuple(info.dim_mathtypes.values()),
				(info.output_shape, info.output_mathtype),
			)
		)

		if valid_viz_modes is None:
			return []

		return valid_viz_modes

	@staticmethod
	def to_plotter(
		value: typ.Self,
	) -> typ.Callable[
		[jtyp.Float32[jtyp.Array, '...'], ct.InfoFlow, mpl_ax.Axis], None
	]:
		return {
			VizMode.Hist1D: image_ops.plot_hist_1d,
			VizMode.BoxPlot1D: image_ops.plot_box_plot_1d,
			VizMode.Curve2D: image_ops.plot_curve_2d,
			VizMode.Points2D: image_ops.plot_points_2d,
			VizMode.Bar: image_ops.plot_bar,
			VizMode.Curves2D: image_ops.plot_curves_2d,
			VizMode.FilledCurves2D: image_ops.plot_filled_curves_2d,
			VizMode.Heatmap2D: image_ops.plot_heatmap_2d,
			# NO PLOTTER: VizMode.SqueezedHeatmap2D
			# NO PLOTTER: VizMode.Heatmap3D
		}[value]

	@staticmethod
	def to_name(value: typ.Self) -> str:
		return {
			VizMode.Hist1D: 'Histogram',
			VizMode.BoxPlot1D: 'Box Plot',
			VizMode.Curve2D: 'Curve',
			VizMode.Points2D: 'Points',
			VizMode.Bar: 'Bar',
			VizMode.Curves2D: 'Curves',
			VizMode.FilledCurves2D: 'Filled Curves',
			VizMode.Heatmap2D: 'Heatmap',
			VizMode.SqueezedHeatmap2D: 'Heatmap (Squeezed)',
			VizMode.Heatmap3D: 'Heatmap (3D)',
		}[value]

	@staticmethod
	def to_icon(value: typ.Self) -> ct.BLIcon:
		return ''


class VizTarget(enum.StrEnum):
	"""Available visualization targets."""

	Plot2D = enum.auto()
	Pixels = enum.auto()
	PixelsPlane = enum.auto()
	Voxels = enum.auto()

	@staticmethod
	def valid_targets_for(viz_mode: VizMode) -> list[typ.Self] | None:
		return {
			VizMode.Hist1D: [VizTarget.Plot2D],
			VizMode.BoxPlot1D: [VizTarget.Plot2D],
			VizMode.Curve2D: [VizTarget.Plot2D],
			VizMode.Points2D: [VizTarget.Plot2D],
			VizMode.Bar: [VizTarget.Plot2D],
			VizMode.Curves2D: [VizTarget.Plot2D],
			VizMode.FilledCurves2D: [VizTarget.Plot2D],
			VizMode.Heatmap2D: [VizTarget.Plot2D, VizTarget.Pixels],
			VizMode.SqueezedHeatmap2D: [VizTarget.Pixels, VizTarget.PixelsPlane],
			VizMode.Heatmap3D: [VizTarget.Voxels],
		}[viz_mode]

	@staticmethod
	def to_name(value: typ.Self) -> str:
		return {
			VizTarget.Plot2D: 'Plot',
			VizTarget.Pixels: 'Pixels',
			VizTarget.PixelsPlane: 'Image Plane',
			VizTarget.Voxels: 'Voxels',
		}[value]

	@staticmethod
	def to_icon(value: typ.Self) -> ct.BLIcon:
		return ''


class VizNode(base.MaxwellSimNode):
	"""Node for visualizing simulation data, by querying its monitors.

	Auto-detects the correct plot type based on the input data:

	Attributes:
		colormap: Colormap to apply to 0..1 output.

	"""

	node_type = ct.NodeType.Viz
	bl_label = 'Viz'
	use_sim_node_name = True

	####################
	# - Sockets
	####################
	input_sockets: typ.ClassVar = {
		'Expr': sockets.ExprSocketDef(
			active_kind=ct.FlowKind.Func,
			default_symbols=[sim_symbols.x],
			default_value=2 * sim_symbols.x.sp_symbol,
		),
	}
	output_sockets: typ.ClassVar = {
		'Preview': sockets.AnySocketDef(),
	}

	managed_obj_types: typ.ClassVar = {
		'plot': managed_objs.ManagedBLImage,
	}

	#####################
	## - Properties
	#####################
	@events.on_value_changed(
		socket_name={'Expr'},
		input_sockets={'Expr'},
		input_socket_kinds={'Expr': ct.FlowKind.Info},
		input_sockets_optional={'Expr': True},
	)
	def on_input_exprs_changed(self, input_sockets) -> None:  # noqa: D102
		has_info = not ct.FlowSignal.check(input_sockets['Expr'])

		info_pending = ct.FlowSignal.check_single(
			input_sockets['Expr'], ct.FlowSignal.FlowPending
		)

		if has_info and not info_pending:
			self.expr_info = bl_cache.Signal.InvalidateCache

	@bl_cache.cached_bl_property()
	def expr_info(self) -> ct.InfoFlow | None:
		info = self._compute_input('Expr', kind=ct.FlowKind.Info)
		if not ct.FlowSignal.check(info):
			return info

		return None

	viz_mode: enum.StrEnum = bl_cache.BLField(
		enum_cb=lambda self, _: self.search_viz_modes(),
		cb_depends_on={'expr_info'},
	)
	viz_target: enum.StrEnum = bl_cache.BLField(
		enum_cb=lambda self, _: self.search_targets(),
		cb_depends_on={'viz_mode'},
	)

	# Mode-Dependent Properties
	colormap: image_ops.Colormap = bl_cache.BLField(
		image_ops.Colormap.Viridis,
	)

	#####################
	## - Searchers
	#####################
	def search_viz_modes(self) -> list[ct.BLEnumElement]:
		if self.expr_info is not None:
			return [
				(
					viz_mode,
					VizMode.to_name(viz_mode),
					VizMode.to_name(viz_mode),
					VizMode.to_icon(viz_mode),
					i,
				)
				for i, viz_mode in enumerate(VizMode.valid_modes_for(self.expr_info))
			]

		return []

	def search_targets(self) -> list[ct.BLEnumElement]:
		if self.viz_mode is not None:
			return [
				(
					viz_target,
					VizTarget.to_name(viz_target),
					VizTarget.to_name(viz_target),
					VizTarget.to_icon(viz_target),
					i,
				)
				for i, viz_target in enumerate(
					VizTarget.valid_targets_for(self.viz_mode)
				)
			]
		return []

	#####################
	## - UI
	#####################
	def draw_label(self):
		if self.viz_mode is not None:
			return 'Viz: ' + self.sim_node_name

		return self.bl_label

	def draw_props(self, _: bpy.types.Context, col: bpy.types.UILayout):
		col.prop(self, self.blfields['viz_mode'], text='')
		col.prop(self, self.blfields['viz_target'], text='')
		if self.viz_target in [VizTarget.Pixels, VizTarget.PixelsPlane]:
			col.prop(self, self.blfields['colormap'], text='')

	####################
	# - Events
	####################
	@events.on_value_changed(
		socket_name='Expr',
		run_on_init=True,
		input_sockets={'Expr'},
		input_socket_kinds={'Expr': {ct.FlowKind.Info, ct.FlowKind.Params}},
		input_sockets_optional={'Expr': True},
	)
	def on_any_changed(self, input_sockets: dict):
		info = input_sockets['Expr'][ct.FlowKind.Info]
		params = input_sockets['Expr'][ct.FlowKind.Params]

		has_info = not ct.FlowSignal.check(info)
		has_params = not ct.FlowSignal.check(params)

		# Provide Sockets for Symbol Realization
		## -> This happens if Params contains not-yet-realized symbols.
		if has_info and has_params and params.symbols:
			if set(self.loose_input_sockets) != {
				sym.name for sym in params.symbols if sym.name in info.dim_names
			}:
				self.loose_input_sockets = {
					sym.name: sockets.ExprSocketDef(
						active_kind=ct.FlowKind.Range,
						size=spux.NumberSize1D.Scalar,
						mathtype=info.dim_mathtypes[sym.name],
						physical_type=info.dim_physical_types[sym.name],
						default_min=(
							info.dim_idx[sym.name].start
							if not sp.S(info.dim_idx[sym.name].start).is_infinite
							else sp.S(0)
						),
						default_max=(
							info.dim_idx[sym.name].start
							if not sp.S(info.dim_idx[sym.name].stop).is_infinite
							else sp.S(1)
						),
						default_steps=50,
					)
					for sym in params.sorted_symbols
					if sym.name in info.dim_names
				}

		elif self.loose_input_sockets:
			self.loose_input_sockets = {}

	#####################
	## - Plotting
	#####################
	@events.computes_output_socket(
		'Preview',
		kind=ct.FlowKind.Value,
		# Loaded
		props={'viz_mode', 'viz_target', 'colormap'},
		input_sockets={'Expr'},
		input_socket_kinds={
			'Expr': {ct.FlowKind.Func, ct.FlowKind.Info, ct.FlowKind.Params}
		},
		all_loose_input_sockets=True,
	)
	def compute_dummy_value(self, props, input_sockets, loose_input_sockets):
		return ct.FlowSignal.NoFlow

	@events.on_show_plot(
		managed_objs={'plot'},
		props={'viz_mode', 'viz_target', 'colormap'},
		input_sockets={'Expr'},
		input_socket_kinds={
			'Expr': {ct.FlowKind.Func, ct.FlowKind.Info, ct.FlowKind.Params}
		},
		unit_systems={'BlenderUnits': ct.UNITS_BLENDER},
		all_loose_input_sockets=True,
		stop_propagation=True,
	)
	def on_show_plot(
		self, managed_objs, props, input_sockets, loose_input_sockets, unit_systems
	):
		# Retrieve Inputs
		lazy_func = input_sockets['Expr'][ct.FlowKind.Func]
		info = input_sockets['Expr'][ct.FlowKind.Info]
		params = input_sockets['Expr'][ct.FlowKind.Params]

		has_info = not ct.FlowSignal.check(info)
		has_params = not ct.FlowSignal.check(params)

		# Invalid Mode | Target
		## -> To limit branching, return now if things aren't right.
		if (
			not has_info
			or not has_params
			or props['viz_mode'] is None
			or props['viz_target'] is None
		):
			return

		# Compute Ranges for Symbols from Loose Sockets
		## -> These are the concrete values of the symbol for plotting.
		## -> In a quite nice turn of events, all this is cached lookups.
		## -> ...Unless something changed, in which case, well. It changed.
		symbol_values = {
			sym: (
				loose_input_sockets[sym.name]
				.realize_array.rescale_to_unit(info.dim_units[sym.name])
				.values
			)
			for sym in params.sorted_symbols
		}

		# Realize Func w/Symbolic Values, Unit System
		## -> This gives us the actual plot data!
		data = lazy_func.func_jax(
			*params.scaled_func_args(
				unit_systems['BlenderUnits'], symbol_values=symbol_values
			),
			**params.scaled_func_kwargs(
				unit_systems['BlenderUnits'], symbol_values=symbol_values
			),
		)

		# Replace InfoFlow Indices w/Realized Symbolic Ranges
		## -> This ensures correct axis scaling.
		if params.symbols:
			info = info.rescale_dim_idxs(loose_input_sockets)

		# Visualize by-Target
		if props['viz_target'] == VizTarget.Plot2D:
			managed_objs['plot'].mpl_plot_to_image(
				lambda ax: VizMode.to_plotter(props['viz_mode'])(data, info, ax),
				bl_select=True,
			)

		if props['viz_target'] == VizTarget.Pixels:
			managed_objs['plot'].map_2d_to_image(
				data,
				colormap=props['colormap'],
				bl_select=True,
			)

		if props['viz_target'] == VizTarget.PixelsPlane:
			raise NotImplementedError

		if props['viz_target'] == VizTarget.Voxels:
			raise NotImplementedError


####################
# - Blender Registration
####################
BL_REGISTER = [
	VizNode,
]
BL_NODES = {ct.NodeType.Viz: (ct.NodeCategory.MAXWELLSIM_ANALYSIS)}
