import bpy
import sympy as sp
import sympy.physics.units as spu
import scipy as sc

from ... import contracts as ct
from ... import sockets
from .. import base

VAC_SPEED_OF_LIGHT = (
	sc.constants.speed_of_light
	* spu.meter/spu.second
)

class WaveConstantNode(base.MaxwellSimNode):
	node_type = ct.NodeType.WaveConstant
	
	bl_label = "Wave Constant"
	
	input_socket_sets = {
		"Vacuum WL": {
			"WL": sockets.PhysicalLengthSocketDef(),
		},
		"Frequency": {
			"Freq": sockets.PhysicalFreqSocketDef(),
		},
	}
	output_sockets = {
		"WL": sockets.PhysicalLengthSocketDef(),
		"Freq": sockets.PhysicalFreqSocketDef(),
	}
	
	####################
	# - Callbacks
	####################
	@base.computes_output_socket(
		"WL",
		kind=ct.DataFlowKind.Value,
		input_sockets={"WL", "Freq"},
	)
	def compute_vac_wl(self, input_socket_values: dict) -> sp.Expr:
		if (vac_wl := input_socket_values["WL"]):
			return vac_wl
		elif (freq := input_socket_values["Freq"]):
			return spu.convert_to(
				VAC_SPEED_OF_LIGHT / freq,
				spu.meter,
			)
		
		raise RuntimeError("Vac WL and Freq are both non-truthy")
	
	@base.computes_output_socket(
		"Freq",
		input_sockets={"WL", "Freq"},
	)
	def compute_freq(self, input_sockets: dict) -> sp.Expr:
		if (vac_wl := input_sockets["WL"]):
			return spu.convert_to(
				VAC_SPEED_OF_LIGHT / vac_wl,
				spu.hertz,
			)
		elif (freq := input_sockets["Freq"]):
			return freq

####################
# - Blender Registration
####################
BL_REGISTER = [
	WaveConstantNode,
]
BL_NODES = {
	ct.NodeType.WaveConstant: (
		ct.NodeCategory.MAXWELLSIM_INPUTS_CONSTANTS
	)
}
