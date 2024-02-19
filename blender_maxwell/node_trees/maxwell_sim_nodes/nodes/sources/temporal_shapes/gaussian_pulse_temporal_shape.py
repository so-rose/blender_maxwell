import tidy3d as td
import sympy as sp
import sympy.physics.units as spu

from .... import contracts
from .... import sockets
from ... import base

class GaussianPulseTemporalShapeNode(base.MaxwellSimTreeNode):
	node_type = contracts.NodeType.GaussianPulseTemporalShape
	
	bl_label = "Gaussian Pulse Temporal Shape"
	#bl_icon = ...
	
	####################
	# - Sockets
	####################
	input_sockets = {
		#"amplitude": sockets.RealNumberSocketDef(
		#	label="Temporal Shape",
		#),  ## Should have a unit of some kind...
		"phase": sockets.PhysicalAngleSocketDef(
			label="Phase",
		),
		"freq_center": sockets.PhysicalFreqSocketDef(
			label="Freq Center",
		),
		"freq_std": sockets.PhysicalFreqSocketDef(
			label="Freq STD",
		),
		"time_delay_rel_ang_freq": sockets.RealNumberSocketDef(
			label="Time Delay rel. Ang. Freq",
		),
		"remove_dc_component": sockets.BoolSocketDef(
			label="Remove DC",
			default_value=True,
		),
	}
	output_sockets = {
		"temporal_shape": sockets.MaxwellTemporalShapeSocketDef(
			label="Temporal Shape",
		),
	}
	
	####################
	# - Output Socket Computation
	####################
	@base.computes_output_socket("temporal_shape")
	def compute_source(self: contracts.NodeTypeProtocol) -> td.PointDipole:
		_phase = self.compute_input("phase")
		_freq_center = self.compute_input("freq_center")
		_freq_std = self.compute_input("freq_std")
		time_delay_rel_ang_freq = self.compute_input("time_delay_rel_ang_freq")
		remove_dc_component = self.compute_input("remove_dc_component")
		
		cheating_amplitude = 1.0
		phase = spu.convert_to(_phase, spu.radian) / spu.radian
		freq_center = spu.convert_to(_freq_center, spu.hertz) / spu.hertz
		freq_std = spu.convert_to(_freq_std, spu.hertz) / spu.hertz
		
		return td.GaussianPulse(
			amplitude=cheating_amplitude,
			phase=phase,
			freq0=freq_center,
			fwidth=freq_std,
			offset=time_delay_rel_ang_freq,
			remove_dc_component=remove_dc_component,
		)



####################
# - Blender Registration
####################
BL_REGISTER = [
	GaussianPulseTemporalShapeNode,
]
BL_NODES = {
	contracts.NodeType.GaussianPulseTemporalShape: (
		contracts.NodeCategory.MAXWELLSIM_SOURCES_TEMPORALSHAPES
	)
}
