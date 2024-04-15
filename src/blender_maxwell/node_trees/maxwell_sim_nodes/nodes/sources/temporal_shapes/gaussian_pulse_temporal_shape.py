import typing as typ

import bpy
import numpy as np
import sympy.physics.units as spu
import tidy3d as td

from ......utils import extra_sympy_units as spuex
from .... import contracts as ct
from .... import managed_objs, sockets
from ... import base, events


class GaussianPulseTemporalShapeNode(base.MaxwellSimNode):
	node_type = ct.NodeType.GaussianPulseTemporalShape
	bl_label = 'Gaussian Pulse Temporal Shape'
	# bl_icon = ...

	####################
	# - Sockets
	####################
	input_sockets = {
		# "amplitude": sockets.RealNumberSocketDef(
		# label="Temporal Shape",
		# ),  ## Should have a unit of some kind...
		'Freq Center': sockets.PhysicalFreqSocketDef(
			default_value=500 * spuex.terahertz,
		),
		'Freq Std.': sockets.PhysicalFreqSocketDef(
			default_value=200 * spuex.terahertz,
		),
		'Phase': sockets.PhysicalAngleSocketDef(),
		'Delay rel. AngFreq': sockets.RealNumberSocketDef(
			default_value=5.0,
		),
		'Remove DC': sockets.BoolSocketDef(
			default_value=True,
		),
	}
	output_sockets = {
		'Temporal Shape': sockets.MaxwellTemporalShapeSocketDef(),
	}

	managed_obj_types = {
		'amp_time': managed_objs.ManagedBLImage,
	}

	####################
	# - Properties
	####################
	plot_time_start: bpy.props.FloatProperty(
		name='Plot Time Start (ps)',
		description='The instance ID of a particular MaxwellSimNode instance, used to index caches',
		default=0.0,
		update=(lambda self, context: self.sync_prop('plot_time_start', context)),
	)
	plot_time_end: bpy.props.FloatProperty(
		name='Plot Time End (ps)',
		description='The instance ID of a particular MaxwellSimNode instance, used to index caches',
		default=5,
		update=(lambda self, context: self.sync_prop('plot_time_start', context)),
	)

	####################
	# - UI
	####################
	def draw_props(self, _, layout):
		layout.label(text='Plot Settings')
		split = layout.split(factor=0.6)

		col = split.column()
		col.label(text='t-Range (ps)')

		col = split.column()
		col.prop(self, 'plot_time_start', text='')
		col.prop(self, 'plot_time_end', text='')

	####################
	# - Output Socket Computation
	####################
	@events.computes_output_socket(
		'Temporal Shape',
		input_sockets={
			'Freq Center',
			'Freq Std.',
			'Phase',
			'Delay rel. AngFreq',
			'Remove DC',
		},
	)
	def compute_source(self, input_sockets: dict) -> td.GaussianPulse:
		if (
			(_freq_center := input_sockets['Freq Center']) is None
			or (_freq_std := input_sockets['Freq Std.']) is None
			or (_phase := input_sockets['Phase']) is None
			or (time_delay_rel_ang_freq := input_sockets['Delay rel. AngFreq']) is None
			or (remove_dc_component := input_sockets['Remove DC']) is None
		):
			raise ValueError('Inputs not defined')

		cheating_amplitude = 1.0
		freq_center = spu.convert_to(_freq_center, spu.hertz) / spu.hertz
		freq_std = spu.convert_to(_freq_std, spu.hertz) / spu.hertz
		phase = spu.convert_to(_phase, spu.radian) / spu.radian

		return td.GaussianPulse(
			amplitude=cheating_amplitude,
			phase=phase,
			freq0=freq_center,
			fwidth=freq_std,
			offset=time_delay_rel_ang_freq,
			remove_dc_component=remove_dc_component,
		)

	@events.on_show_plot(
		managed_objs={'amp_time'},
		props={'plot_time_start', 'plot_time_end'},
		output_sockets={'Temporal Shape'},
		stop_propagation=True,
	)
	def on_show_plot(
		self,
		managed_objs: dict,
		output_sockets: dict,
		props: dict,
	):
		temporal_shape = output_sockets['Temporal Shape']
		plot_time_start = props['plot_time_start'] * 1e-15
		plot_time_end = props['plot_time_end'] * 1e-15

		times = np.linspace(plot_time_start, plot_time_end)

		managed_objs['amp_time'].mpl_plot_to_image(
			lambda ax: temporal_shape.plot_spectrum(times, ax=ax),
			bl_select=True,
		)


####################
# - Blender Registration
####################
BL_REGISTER = [
	GaussianPulseTemporalShapeNode,
]
BL_NODES = {
	ct.NodeType.GaussianPulseTemporalShape: (
		ct.NodeCategory.MAXWELLSIM_SOURCES_TEMPORALSHAPES
	)
}
