import tidy3d as td
import sympy as sp
import sympy.physics.units as spu

from .... import contracts
from .... import sockets
from ... import base


class ContinuousWaveTemporalShapeNode(base.MaxwellSimTreeNode):
	node_type = contracts.NodeType.ContinuousWaveTemporalShape

	bl_label = 'Continuous Wave Temporal Shape'
	# bl_icon = ...

	####################
	# - Sockets
	####################
	input_sockets = {
		# "amplitude": sockets.RealNumberSocketDef(
		# label="Temporal Shape",
		# ),  ## Should have a unit of some kind...
		'phase': sockets.PhysicalAngleSocketDef(
			label='Phase',
		),
		'freq_center': sockets.PhysicalFreqSocketDef(
			label='Freq Center',
		),
		'freq_std': sockets.PhysicalFreqSocketDef(
			label='Freq STD',
		),
		'time_delay_rel_ang_freq': sockets.RealNumberSocketDef(
			label='Time Delay rel. Ang. Freq',
			default_value=5.0,
		),
	}
	output_sockets = {
		'temporal_shape': sockets.MaxwellTemporalShapeSocketDef(
			label='Temporal Shape',
		),
	}

	####################
	# - Output Socket Computation
	####################
	@base.computes_output_socket('temporal_shape')
	def compute_source(self: contracts.NodeTypeProtocol) -> td.PointDipole:
		_phase = self.compute_input('phase')
		_freq_center = self.compute_input('freq_center')
		_freq_std = self.compute_input('freq_std')
		time_delay_rel_ang_freq = self.compute_input('time_delay_rel_ang_freq')

		cheating_amplitude = 1.0
		phase = spu.convert_to(_phase, spu.radian) / spu.radian
		freq_center = spu.convert_to(_freq_center, spu.hertz) / spu.hertz
		freq_std = spu.convert_to(_freq_std, spu.hertz) / spu.hertz

		return td.ContinuousWave(
			amplitude=cheating_amplitude,
			phase=phase,
			freq0=freq_center,
			fwidth=freq_std,
			offset=time_delay_rel_ang_freq,
		)


####################
# - Blender Registration
####################
BL_REGISTER = [
	ContinuousWaveTemporalShapeNode,
]
BL_NODES = {
	contracts.NodeType.ContinuousWaveTemporalShape: (
		contracts.NodeCategory.MAXWELLSIM_SOURCES_TEMPORALSHAPES
	)
}
