import bpy
import sympy as sp
import sympy.physics.units as spu
import scipy as sc

from .....utils import extra_sympy_units as spux
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
		# Single
		"Vacuum WL": {
			"WL": sockets.PhysicalLengthSocketDef(
				default_value=500*spu.nm,
				default_unit=spu.nm,
			),
		},
		"Frequency": {
			"Freq": sockets.PhysicalFreqSocketDef(
				default_value=500*spux.THz,
				default_unit=spux.THz,
			),
		},
		
		# Listy
		"Vacuum WLs": {
			"WLs": sockets.PhysicalLengthSocketDef(
				is_list=True,
			),
		},
		"Frequencies": {
			"Freqs": sockets.PhysicalFreqSocketDef(
				is_list=True,
			),
		},
	}
	
	####################
	# - Callbacks
	####################
	@base.computes_output_socket(
		"WL",
		input_sockets={"WL", "Freq"},
	)
	def compute_vac_wl(self, input_sockets: dict) -> sp.Expr:
		if (vac_wl := input_sockets["WL"]) is not None:
			return vac_wl
		
		elif (freq := input_sockets["Freq"]) is not None:
			return spu.convert_to(
				VAC_SPEED_OF_LIGHT / freq,
				spu.meter,
			)
		
		raise RuntimeError("Vac WL and Freq are both None")
	
	@base.computes_output_socket(
		"Freq",
		input_sockets={"WL", "Freq"},
	)
	def compute_freq(self, input_sockets: dict) -> sp.Expr:
		if (vac_wl := input_sockets["WL"]) is not None:
			return spu.convert_to(
				VAC_SPEED_OF_LIGHT / vac_wl,
				spu.hertz,
			)
		elif (freq := input_sockets["Freq"]) is not None:
			return freq
		
		raise RuntimeError("Vac WL and Freq are both None")
	
	####################
	# - Listy Callbacks
	####################
	@base.computes_output_socket(
		"WLs",
		input_sockets={"WLs", "Freqs"},
	)
	def compute_vac_wls(self, input_sockets: dict) -> sp.Expr:
		if (vac_wls := input_sockets["WLs"]) is not None:
			return vac_wls
		elif (freqs := input_sockets["Freqs"]) is not None:
			return [
				spu.convert_to(
					VAC_SPEED_OF_LIGHT / freq,
					spu.meter,
				)
				for freq in freqs
			][::-1]
		
		raise RuntimeError("Vac WLs and Freqs are both None")
	
	@base.computes_output_socket(
		"Freqs",
		input_sockets={"WLs", "Freqs"},
	)
	def compute_freqs(self, input_sockets: dict) -> sp.Expr:
		if (vac_wls := input_sockets["WLs"]) is not None:
			return [
				spu.convert_to(
					VAC_SPEED_OF_LIGHT / vac_wl,
					spu.hertz,
				)
				for vac_wl in vac_wls
			][::-1]
		elif (freqs := input_sockets["Freqs"]) is not None:
			return freqs
		
		raise RuntimeError("Vac WLs and Freqs are both None")
	
	####################
	# - Callbacks
	####################
	@base.on_value_changed(
		prop_name="active_socket_set",
		props={"active_socket_set"}
	)
	def on_value_changed__active_socket_set(self, props: dict):
		# Singular: Normal Output Sockets
		if props["active_socket_set"] in {"Vacuum WL", "Frequency"}:
			self.loose_output_sockets = {}
			self.loose_output_sockets = {
				"Freq": sockets.PhysicalFreqSocketDef(),
				"WL": sockets.PhysicalLengthSocketDef(),
			}
		
		# Plural: Listy Output Sockets
		elif props["active_socket_set"] in {"Vacuum WLs", "Frequencies"}:
			self.loose_output_sockets = {}
			self.loose_output_sockets = {
				"Freqs": sockets.PhysicalFreqSocketDef(is_list=True),
				"WLs": sockets.PhysicalLengthSocketDef(is_list=True),
			}
		
		else:
			msg = f"Active socket set invalid for wave constant: {props['active_socket_set']}"
			raise RuntimeError(msg)
	
	@base.on_init()
	def on_init(self):
		self.on_value_changed__active_socket_set()
	

####################
# - Blender Registration
####################
BL_REGISTER = [
	WaveConstantNode,
]
BL_NODES = {
	ct.NodeType.WaveConstant: (
		ct.NodeCategory.MAXWELLSIM_INPUTS
	)
}
