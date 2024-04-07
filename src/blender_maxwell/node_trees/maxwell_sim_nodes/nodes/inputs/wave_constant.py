import typing as typ

import sympy as sp
import sympy.physics.units as spu

from .....utils import extra_sympy_units as spux
from .....utils import sci_constants as constants
from ... import contracts as ct
from ... import sockets
from .. import base, events


class WaveConstantNode(base.MaxwellSimNode):
	node_type = ct.NodeType.WaveConstant
	bl_label = 'Wave Constant'

	input_socket_sets: typ.ClassVar = {
		# Single
		'Vacuum WL': {
			'WL': sockets.PhysicalLengthSocketDef(
				default_value=500 * spu.nm,
				default_unit=spu.nm,
			),
		},
		'Frequency': {
			'Freq': sockets.PhysicalFreqSocketDef(
				default_value=500 * spux.THz,
				default_unit=spux.THz,
			),
		},
		# Listy
		'Vacuum WLs': {
			'WLs': sockets.PhysicalLengthSocketDef(
				is_list=True,
			),
		},
		'Frequencies': {
			'Freqs': sockets.PhysicalFreqSocketDef(
				is_list=True,
			),
		},
	}

	####################
	# - Event Methods: Listy Output
	####################
	@events.computes_output_socket(
		'WL',
		input_sockets={'WL'},
	)
	def compute_vacwl_from_vacwl(self, input_sockets: dict) -> sp.Expr:
		return input_sockets['WL']

	@events.computes_output_socket(
		'WL',
		input_sockets={'Freq'},
	)
	def compute_freq_from_vacwl(self, input_sockets: dict) -> sp.Expr:
		return constants.vac_speed_of_light / input_sockets['Freq']

	####################
	# - Event Methods: Listy Output
	####################
	@events.computes_output_socket(
		'WLs',
		input_sockets={'WLs', 'Freqs'},
	)
	def compute_vac_wls(self, input_sockets: dict) -> sp.Expr:
		if (vac_wls := input_sockets['WLs']) is not None:
			return vac_wls
		if (freqs := input_sockets['Freqs']) is not None:
			return [constants.vac_speed_of_light / freq for freq in freqs][::-1]

		msg = 'Vac WL and Freq are both None'
		raise RuntimeError(msg)

	@events.computes_output_socket(
		'Freqs',
		input_sockets={'WLs', 'Freqs'},
	)
	def compute_freqs(self, input_sockets: dict) -> sp.Expr:
		if (vac_wls := input_sockets['WLs']) is not None:
			return [constants.vac_speed_of_light / vac_wl for vac_wl in vac_wls][::-1]
		if (freqs := input_sockets['Freqs']) is not None:
			return freqs

		msg = 'Vac WL and Freq are both None'
		raise RuntimeError(msg)

	####################
	# - Event Methods
	####################
	@events.on_value_changed(prop_name='active_socket_set', props={'active_socket_set'})
	def on_active_socket_set_changed(self, props: dict):
		# Singular: Normal Output Sockets
		if props['active_socket_set'] in {'Vacuum WL', 'Frequency'}:
			self.loose_output_sockets = {}
			self.loose_output_sockets = {
				'Freq': sockets.PhysicalFreqSocketDef(),
				'WL': sockets.PhysicalLengthSocketDef(),
			}

		# Plural: Listy Output Sockets
		elif props['active_socket_set'] in {'Vacuum WLs', 'Frequencies'}:
			self.loose_output_sockets = {}
			self.loose_output_sockets = {
				'Freqs': sockets.PhysicalFreqSocketDef(is_list=True),
				'WLs': sockets.PhysicalLengthSocketDef(is_list=True),
			}

		else:
			msg = f"Active socket set invalid for wave constant: {props['active_socket_set']}"
			raise RuntimeError(msg)

	@events.on_init()
	def on_init(self):
		self.on_active_socket_set_changed()


####################
# - Blender Registration
####################
BL_REGISTER = [
	WaveConstantNode,
]
BL_NODES = {ct.NodeType.WaveConstant: (ct.NodeCategory.MAXWELLSIM_INPUTS)}
