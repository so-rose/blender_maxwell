import enum
import typing as typ

import bpy
import jaxtyping as jtyp
import matplotlib.axis as mpl_ax

from blender_maxwell.utils import bl_cache, image_ops, logger
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
			None: [],
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
		'Expr': sockets.ExprSocketDef(),
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
	viz_mode: enum.Enum = bl_cache.BLField(
		prop_ui=True, enum_cb=lambda self, _: self.search_modes()
	)
	viz_target: enum.Enum = bl_cache.BLField(
		prop_ui=True, enum_cb=lambda self, _: self.search_targets()
	)

	# Mode-Dependent Properties
	colormap: image_ops.Colormap = bl_cache.BLField(
		image_ops.Colormap.Viridis, prop_ui=True
	)

	#####################
	## - Mode Searcher
	#####################
	@property
	def data_info(self) -> ct.InfoFlow:
		return self._compute_input('Expr', kind=ct.FlowKind.Info)

	def search_modes(self) -> list[ct.BLEnumElement]:
		if not ct.FlowSignal.check(self.data_info):
			return [
				(
					viz_mode,
					VizMode.to_name(viz_mode),
					VizMode.to_name(viz_mode),
					VizMode.to_icon(viz_mode),
					i,
				)
				for i, viz_mode in enumerate(VizMode.valid_modes_for(self.data_info))
			]

		return []

	#####################
	## - Target Searcher
	#####################
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
		input_sockets={'Expr'},
		run_on_init=True,
		input_socket_kinds={'Expr': ct.FlowKind.Info},
		input_sockets_optional={'Expr': True},
	)
	def on_any_changed(self, input_sockets: dict):
		if not ct.FlowSignal.check_single(
			input_sockets['Expr'], ct.FlowSignal.FlowPending
		):
			self.viz_mode = bl_cache.Signal.ResetEnumItems
			self.viz_target = bl_cache.Signal.ResetEnumItems

	@events.on_value_changed(
		prop_name='viz_mode',
		run_on_init=True,
	)
	def on_viz_mode_changed(self):
		self.viz_target = bl_cache.Signal.ResetEnumItems

	#####################
	## - Plotting
	#####################
	@events.on_show_plot(
		managed_objs={'plot'},
		props={'viz_mode', 'viz_target', 'colormap'},
		input_sockets={'Expr'},
		input_socket_kinds={'Expr': {ct.FlowKind.Array, ct.FlowKind.Info}},
		stop_propagation=True,
	)
	def on_show_plot(
		self,
		managed_objs: dict,
		input_sockets: dict,
		props: dict,
	):
		# Retrieve Inputs
		array_flow = input_sockets['Expr'][ct.FlowKind.Array]
		info = input_sockets['Expr'][ct.FlowKind.Info]

		# Check Flow
		if (
			any(ct.FlowSignal.check(inp) for inp in [array_flow, info])
			or props['viz_mode'] is None
			or props['viz_target'] is None
		):
			return

		# Viz Target
		if props['viz_target'] == VizTarget.Plot2D:
			managed_objs['plot'].mpl_plot_to_image(
				lambda ax: VizMode.to_plotter(props['viz_mode'])(
					array_flow.values, info, ax
				),
				bl_select=True,
			)
		if props['viz_target'] == VizTarget.Pixels:
			managed_objs['plot'].map_2d_to_image(
				array_flow.values,
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
