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
import sympy.physics.units as spu

from blender_maxwell.utils import bl_cache, image_ops, logger, sim_symbols
from blender_maxwell.utils import sympy_extra as spux

from ... import contracts as ct
from ... import managed_objs, sockets
from .. import base, events

log = logger.get(__name__)


class VizMode(enum.StrEnum):
	"""Available visualization modes.

	**NOTE**: >1D output dimensions currently have no viz.

	Plots for `() -> ℝ`:
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

	BoxPlot1D = enum.auto()

	Curve2D = enum.auto()
	Points2D = enum.auto()
	Bar = enum.auto()

	Curves2D = enum.auto()
	FilledCurves2D = enum.auto()

	Heatmap2D = enum.auto()

	SqueezedHeatmap2D = enum.auto()
	Heatmap3D = enum.auto()

	####################
	# - UI
	####################
	@staticmethod
	def to_name(value: typ.Self) -> str:
		return {
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

	####################
	# - Validity
	####################
	@staticmethod
	def by_info(info: ct.InfoFlow) -> list[typ.Self] | None:
		"""Given the input `InfoFlow`, deduce which visualization modes are valid to use with the described data."""
		Z = spux.MathType.Integer
		R = spux.MathType.Real
		C = spux.MathType.Complex
		VM = VizMode

		return {
			((Z), (1, 1, R)): [
				VM.BoxPlot1D,
			],
			((R,), (1, 1, R)): [
				VM.Curve2D,
				VM.Points2D,
				VM.Bar,
			],
			((R,), (1, 1, C)): [
				VM.Curve2D,
			],
			((R, Z), (1, 1, R)): [
				VM.Curves2D,
				VM.FilledCurves2D,
			],
			((R, R), (1, 1, R)): [
				VM.Heatmap2D,
			],
			((R, R, R), (1, 1, R)): [
				VM.SqueezedHeatmap2D,
				VM.Heatmap3D,
			],
		}.get(
			(
				tuple([dim.mathtype for dim in info.dims]),
				(info.output.rows, info.output.cols, info.output.mathtype),
			),
			[],
		)

	####################
	# - Properties
	####################
	@property
	def mpl_plotter(
		self,
	) -> typ.Callable[
		[jtyp.Float32[jtyp.Array, '...'], ct.InfoFlow, mpl_ax.Axis], None
	]:
		return {
			VizMode.BoxPlot1D: image_ops.plot_box_plot_1d,
			VizMode.Curve2D: image_ops.plot_curve_2d,
			VizMode.Points2D: image_ops.plot_points_2d,
			VizMode.Bar: image_ops.plot_bar,
			VizMode.Curves2D: image_ops.plot_curves_2d,
			VizMode.FilledCurves2D: image_ops.plot_filled_curves_2d,
			VizMode.Heatmap2D: image_ops.plot_heatmap_2d,
			# NO PLOTTER: VizMode.SqueezedHeatmap2D
			# NO PLOTTER: VizMode.Heatmap3D
		}[self]


class VizTarget(enum.StrEnum):
	"""Available visualization targets."""

	Plot2D = enum.auto()
	Pixels = enum.auto()
	PixelsPlane = enum.auto()
	Voxels = enum.auto()

	@staticmethod
	def valid_targets_for(viz_mode: VizMode) -> list[typ.Self] | None:
		return {
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


sym_x_um = sim_symbols.space_x(spu.um)
x_um = sym_x_um.sp_symbol


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
			default_symbols=[sym_x_um],
			default_value=sp.exp(-(x_um**2)),
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
		# Trigger
		socket_name={'Expr'},
		# Loaded
		input_sockets={'Expr'},
		input_socket_kinds={'Expr': ct.FlowKind.Info},
		input_sockets_optional={'Expr': True},
		# Flow
		## -> See docs in TransformMathNode
		stop_propagation=True,
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

	viz_mode: VizMode = bl_cache.BLField(
		enum_cb=lambda self, _: self.search_viz_modes(),
		cb_depends_on={'expr_info'},
	)
	viz_target: VizTarget = bl_cache.BLField(
		enum_cb=lambda self, _: self.search_targets(),
		cb_depends_on={'viz_mode'},
	)

	# Plot
	plot_width: float = bl_cache.BLField(6.0, abs_min=0.1)
	plot_height: float = bl_cache.BLField(3.0, abs_min=0.1)
	plot_dpi: int = bl_cache.BLField(150, abs_min=25)

	# Pixels
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
				for i, viz_mode in enumerate(VizMode.by_info(self.expr_info))
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

		if self.viz_target is VizTarget.Plot2D:
			row = col.row(align=True)
			row.alignment = 'CENTER'
			row.label(text='Width | Height | DPI')

			row = col.row(align=True)
			row.prop(self, self.blfields['plot_width'], text='')
			row.prop(self, self.blfields['plot_height'], text='')

			row = col.row(align=True)
			col.prop(self, self.blfields['plot_dpi'], text='')

	####################
	# - Events
	####################
	@events.on_value_changed(
		# Trigger
		socket_name='Expr',
		run_on_init=True,
		# Loaded
		input_sockets={'Expr'},
		input_socket_kinds={'Expr': {ct.FlowKind.Info, ct.FlowKind.Params}},
		input_sockets_optional={'Expr': True},
	)
	def on_any_changed(self, input_sockets: dict):
		info = input_sockets['Expr'][ct.FlowKind.Info]
		params = input_sockets['Expr'][ct.FlowKind.Params]

		has_info = not ct.FlowSignal.check(info)
		has_params = not ct.FlowSignal.check(params)

		# Declare Loose Sockets that Realize Symbols
		## -> This happens if Params contains not-yet-realized symbols.
		if has_info and has_params and params.symbols:
			if set(self.loose_input_sockets) != {sym.name for sym in params.symbols}:
				self.loose_input_sockets = {
					sym.name: sockets.ExprSocketDef(
						**(
							expr_info
							| {
								'active_kind': ct.FlowKind.Range
								if sym in info.dims
								else ct.FlowKind.Value
							}
						)
					)
					for sym, expr_info in params.sym_expr_infos.items()
				}

		elif self.loose_input_sockets:
			self.loose_input_sockets = {}

	#####################
	## - FlowKind.Value
	#####################
	@events.computes_output_socket(
		'Preview',
		kind=ct.FlowKind.Previews,
		# Loaded
		props={
			'sim_node_name',
			'viz_mode',
			'viz_target',
			'colormap',
			'plot_width',
			'plot_height',
			'plot_dpi',
		},
		input_sockets={'Expr'},
		input_socket_kinds={
			'Expr': {ct.FlowKind.Func, ct.FlowKind.Info, ct.FlowKind.Params}
		},
		all_loose_input_sockets=True,
	)
	def compute_previews(self, props, input_sockets, loose_input_sockets):
		"""Needed for the plot to regenerate in the viewer."""
		return ct.PreviewsFlow(bl_image_name=props['sim_node_name'])

	#####################
	## - On Show Plot
	#####################
	@events.on_show_plot(
		managed_objs={'plot'},
		props={
			'viz_mode',
			'viz_target',
			'colormap',
			'plot_width',
			'plot_height',
			'plot_dpi',
		},
		input_sockets={'Expr'},
		input_socket_kinds={
			'Expr': {ct.FlowKind.Func, ct.FlowKind.Info, ct.FlowKind.Params}
		},
		all_loose_input_sockets=True,
		stop_propagation=True,
	)
	def on_show_plot(
		self, managed_objs, props, input_sockets, loose_input_sockets
	) -> None:
		log.debug('Show Plot')
		lazy_func = input_sockets['Expr'][ct.FlowKind.Func]
		info = input_sockets['Expr'][ct.FlowKind.Info]
		params = input_sockets['Expr'][ct.FlowKind.Params]

		has_info = not ct.FlowSignal.check(info)
		has_params = not ct.FlowSignal.check(params)

		plot = managed_objs['plot']
		viz_mode = props['viz_mode']
		viz_target = props['viz_target']
		if has_info and has_params and viz_mode is not None and viz_target is not None:
			# Retrieve Data
			## -> The loose input socket values are user-selected symbol values.
			## -> These are used to get rid of symbols in the ParamsFlow.
			## -> What's left is a dictionary from SimSymbol -> Data
			data = lazy_func.realize_as_data(
				info,
				params,
				symbol_values={
					sym: loose_input_sockets[sym.name] for sym in params.sorted_symbols
				},
			)
			## TODO: CACHE entries that don't change, PLEASEEE

			# Match Viz Type & Perform Visualization
			## -> Viz Target determines how to plot.
			## -> Viz Mode may help select a particular plotting method.
			## -> Other parameters may be uses, depending on context.
			match viz_target:
				case VizTarget.Plot2D:
					plot_width = props['plot_width']
					plot_height = props['plot_height']
					plot_dpi = props['plot_dpi']
					plot.mpl_plot_to_image(
						lambda ax: viz_mode.mpl_plotter(data, ax),
						width_inches=plot_width,
						height_inches=plot_height,
						dpi=plot_dpi,
					)

				case VizTarget.Pixels:
					colormap = props['colormap']
					if colormap is not None:
						plot.map_2d_to_image(
							data,
							colormap=colormap,
						)

				case VizTarget.PixelsPlane:
					raise NotImplementedError

				case VizTarget.Voxels:
					raise NotImplementedError


####################
# - Blender Registration
####################
BL_REGISTER = [
	VizNode,
]
BL_NODES = {ct.NodeType.Viz: (ct.NodeCategory.MAXWELLSIM_ANALYSIS)}
