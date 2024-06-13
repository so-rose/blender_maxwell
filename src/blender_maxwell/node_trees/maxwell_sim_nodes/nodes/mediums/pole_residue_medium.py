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

import typing as typ

import bpy
import sympy.physics.units as spu
import tidy3d as td
import tidy3d.plugins.dispersion as td_dispersion

from blender_maxwell.utils import bl_cache, logger
from blender_maxwell.utils import sympy_extra as spux

from ... import contracts as ct
from ... import managed_objs, sockets
from .. import base, events

log = logger.get(__name__)

FK = ct.FlowKind
FS = ct.FlowSignal
PT = spux.PhysicalType

VALID_URL_PREFIXES = {
	'https://refractiveindex.info',
}


####################
# - Operators
####################
class FitPoleResidueMedium(bpy.types.Operator):
	"""Trigger fitting of a dispersive medium to a Pole-Residue model, and store it on a `PoleResidueMediumnode`."""

	bl_idname = ct.OperatorType.NodeFitDispersiveMedium
	bl_label = 'Fit Dispersive Medium from Input'
	bl_description = (
		'Fit the dispersive medium specified by the `PoleResidueMediumNode`.'
	)

	@classmethod
	def poll(cls, context):
		return (
			# Check Tidy3DWebExporter is Accessible
			hasattr(context, 'node')
			and hasattr(context.node, 'node_type')
			and context.node.node_type == ct.NodeType.PoleResidueMedium
			# Check Medium is Fittable
			and context.node.fitter is not None
			and context.node.fitted_medium is None
			and context.node.fitted_rms_error is None
		)

	def execute(self, context):
		node = context.node

		try:
			pole_residue_medium, rms_error = node.fitter.fit(
				min_num_poles=node.fit_min_poles,
				max_num_poles=node.fit_max_poles,
				tolerance_rms=node.fit_tolerance_rms,
			)

		except:  # noqa: E722
			self.report(
				{'ERROR'}, "Couldn't perform PoleResidue data fit - check inputs."
			)
			return {'FINISHED'}

		else:
			node.fitted_medium = pole_residue_medium
			node.fitted_rms_error = float(rms_error)
			for bl_socket in node.inputs:
				bl_socket.trigger_event(ct.FlowEvent.EnableLock)

		return {'FINISHED'}


class ReleasePoleResidueFit(bpy.types.Operator):
	"""Release a previous fit of a dispersive medium to a Pole-Residue model, from a `PoleResidueMediumnode`."""

	bl_idname = ct.OperatorType.NodeReleaseDispersiveFit
	bl_label = 'Release Dispersive Medium fit'
	bl_description = (
		'Release the dispersive medium fit from the `PoleResidueMediumNode`.'
	)

	@classmethod
	def poll(cls, context):
		return (
			# Check Tidy3DWebExporter is Accessible
			hasattr(context, 'node')
			and hasattr(context.node, 'node_type')
			and context.node.node_type == ct.NodeType.PoleResidueMedium
			# Check Medium is Fittable
			and context.node.fitted_medium is not None
			and context.node.fitted_rms_error is not None
		)

	def execute(self, context):
		node = context.node

		node.fitted_medium = None
		node.fitted_rms_error = None
		for bl_socket in node.inputs:
			bl_socket.trigger_event(ct.FlowEvent.DisableLock)

		return {'FINISHED'}


####################
# - Node
####################
class PoleResidueMediumNode(base.MaxwellSimNode):
	"""A dispersive medium described by a pole-residue model."""

	node_type = ct.NodeType.PoleResidueMedium
	bl_label = 'Pole Residue Medium'

	input_socket_sets: typ.ClassVar = {
		'Fit URL': {
			'URL': sockets.StringSocketDef(),
		},
		'Fit Data': {
			'Expr': sockets.ExprSocketDef(active_kind=FK.Func),
		},
	}
	output_sockets: typ.ClassVar = {
		'Medium': sockets.MaxwellMediumSocketDef(),
	}
	managed_obj_types: typ.ClassVar = {
		'plot': managed_objs.ManagedBLImage,
	}

	####################
	# - Properties
	####################
	fit_min_poles: int = bl_cache.BLField(1)
	fit_max_poles: int = bl_cache.BLField(5)
	fit_tolerance_rms: float = bl_cache.BLField(0.001)

	fitted_medium: td.PoleResidue | None = bl_cache.BLField(None)
	fitted_rms_error: float | None = bl_cache.BLField(None)

	## TODO: Bool of whether to fit eps_inf, with conditional choice of eps_inf as socket
	## TODO: "AdvanceFastFitterParam" options incl. loss_bounds, weights, show_progress, show_unweighted_rms, relaxed, smooth, logspacing, numiters, passivity_num_iters, and slsqp_constraint_scale

	####################
	# - Data Fitting
	####################
	@events.on_value_changed(
		socket_name={'Expr': {FK.Func, FK.Params, FK.Info}, 'URL': FK.Value},
		stop_propagation=True,
	)
	def on_expr_changed(self) -> None:
		"""Respond to changes in `Func`, `Params`, and `Info` to invalidate `self.fitter`."""
		self.fitter = bl_cache.Signal.InvalidateCache

	@bl_cache.cached_bl_property(depends_on={'active_socket_set'})
	def fitter(self) -> td_dispersion.FastDispersionFitter | None:
		"""Compute a `FastDispersionFitter`, which can be used to initiate a data-fit."""
		match self.active_socket_set:
			case 'Fit Data':
				func = self._compute_input('Expr', kind=FK.Func)
				info = self._compute_input('Expr', kind=FK.Info)
				params = self._compute_input('Expr', kind=FK.Params)

				has_info = not FS.check(info)

				expr = events.realize_known({FK.Func: func, FK.Params: params})
				if (
					expr is not None
					and has_info
					and len(info.dims == 1)
					and info.first_dim
					and info.first_dim.physical_type is PT.Length
				):
					return td_dispersion.FastDispersionFitter(
						wvl_um=info.dims[info.first_dim]
						.rescale_to_unit(spu.micrometer)
						.values,
						n_data=expr.real,
						k_data=expr.imag,
					)
				return None

			case 'Fit URL':
				url = self._compute_input('URL', kind=FK.Value)
				has_url = not FS.check(url)

				if has_url and any(
					url.startswith(valid_prefix) for valid_prefix in VALID_URL_PREFIXES
				):
					return None
					# return td_dispersion.FastDispersionFitter.from_url(url)
				return None

		raise TypeError

	####################
	# - UI
	####################
	def draw_props(self, _: bpy.types.Context, col: bpy.types.UILayout):
		"""Draw loaded properties."""
		# Fit/Release Operator
		row = col.row(align=True)
		row.operator(
			ct.OperatorType.NodeFitDispersiveMedium,
			text='Fit Medium',
		)
		if self.fitted_medium is not None:
			row.operator(
				ct.OperatorType.NodeReleaseDispersiveFit,
				icon='LOOP_BACK',
				text='',
			)

		# Fit Parameters / Fit Info
		if self.fitted_medium is None:
			row = col.row(align=True)
			row.alignment = 'CENTER'
			row.label(text='min|max|tol')

			col.prop(self, self.blfields['fit_min_poles'])
			col.prop(self, self.blfields['fit_max_poles'])
			col.prop(self, self.blfields['fit_tolerance_rms'])
		else:
			box = col.box()
			row = box.row(align=True)
			row.alignment = 'CENTER'
			row.label(text='Fit Info')

			split = box.split(factor=0.4)

			col = split.column()
			row = col.row()
			row.label(text='RMS Err')

			col = split.column()
			row = col.row()
			row.label(text=f'{self.fitted_rms_error:.4f}')

	####################
	# - FlowKind.Value
	####################
	@events.computes_output_socket(
		'Medium',
		kind=FK.Value,
		# Loaded
		props={'fitted_medium'},
	)
	def compute_fitted_medium_value(self, props) -> td.Medium | FS:
		"""Return the fitted medium."""
		fitted_medium = props['fitted_medium']
		if fitted_medium is not None:
			return fitted_medium
		return FS.FlowPending

	####################
	# - FlowKind.Func
	####################
	@events.computes_output_socket(
		'Medium',
		kind=FK.Func,
		# Loaded
		outscks_kinds={'Medium': FK.Value},
	)
	def compute_fitted_medium_func(self, output_sockets) -> td.Medium | FS:
		"""Return the fitted medium as a function with that medium baked in."""
		fitted_medium = output_sockets['Medium']
		return ct.FuncFlow(
			func=lambda: fitted_medium,
		)

	####################
	# - FlowKind.Func
	####################
	@events.computes_output_socket(
		'Medium',
		kind=FK.Params,
	)
	def compute_fitted_medium_params(self) -> td.Medium | FS:
		"""Declare no function parameters."""
		return ct.ParamsFlow()

	####################
	# - Event Methods: Plot
	####################
	@events.on_show_plot(
		managed_objs={'plot'},
		# Loaded
		props={'fitter', 'fitted_medium'},
	)
	def on_show_plot(self, props, managed_objs):
		"""When the filetype is 'Experimental Dispersive Medium', plot the computed model against the input data."""
		fitter = props['fitter']
		fitted_medium = props['fitted_medium']
		if fitter is not None and fitted_medium is not None:
			managed_objs['plot'].mpl_plot_to_image(
				lambda ax: props['fitter'].plot(
					medium=props['fitted_medium'],
					ax=ax,
				),
				width_inches=6.0,
				height_inches=3.0,
				dpi=150,
			)
		return FS.FlowPending


####################
# - Blender Registration
####################
BL_REGISTER = [
	FitPoleResidueMedium,
	ReleasePoleResidueFit,
	PoleResidueMediumNode,
]
BL_NODES = {ct.NodeType.PoleResidueMedium: (ct.NodeCategory.MAXWELLSIM_MEDIUMS)}
