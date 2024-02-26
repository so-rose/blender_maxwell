import bpy
import sympy as sp
import sympy.physics.units as spu
import scipy as sc

from .... import contracts
from .... import sockets
from ... import base

vac_speed_of_light = (
	sc.constants.speed_of_light
	* spu.meter/spu.second
)

class WaveConstantNode(base.MaxwellSimTreeNode):
	node_type = contracts.NodeType.WaveConstant
	
	bl_label = "Wave Constant"
	
	input_sockets = {}
	input_socket_sets = {
		"vac_wl": {
			"vac_wl": sockets.PhysicalVacWLSocketDef(
				label="Vac WL",
			),
		},
		"freq": {
			"freq": sockets.PhysicalFreqSocketDef(
				label="Freq",
			),
		},
	}
	output_sockets = {
		"vac_wl": sockets.PhysicalVacWLSocketDef(
			label="Vac WL",
		),
		"freq": sockets.PhysicalVacWLSocketDef(
			label="Freq",
		),
	}
	output_socket_sets = {}
	
	####################
	# - Callbacks
	####################
	@base.computes_output_socket("vac_wl")
	def compute_vac_wl(self: contracts.NodeTypeProtocol) -> sp.Expr:
		if self.socket_set == "vac_wl":
			return self.compute_input("vac_wl")
			
		elif self.socket_set == "freq":
			freq = self.compute_input("freq")
			return spu.convert_to(
				vac_speed_of_light / freq,
				spu.meter,
			)
		
		raise ValueError("No valid socket set.")
	
	@base.computes_output_socket("freq")
	def compute_freq(self: contracts.NodeTypeProtocol) -> sp.Expr:
		if self.socket_set == "vac_wl":
			vac_wl = self.compute_input("vac_wl")
			return spu.convert_to(
				vac_speed_of_light / vac_wl,
				spu.hertz,
			)
			
		elif self.socket_set == "freq":
			return self.compute_input("freq")
		
		raise ValueError("No valid socket set.")



####################
# - Blender Registration
####################
BL_REGISTER = [
	WaveConstantNode,
]
BL_NODES = {
	contracts.NodeType.WaveConstant: (
		contracts.NodeCategory.MAXWELLSIM_INPUTS_CONSTANTS
	)
}
