import tidy3d as td
import sympy as sp
import sympy.physics.units as spu
import scipy as sc

from .... import contracts
from .... import sockets
from ... import base


class WaveConverterNode(base.MaxwellSimTreeNode):
	node_type = contracts.NodeType.WaveConverter
	bl_label = 'Wave Converter'
	# bl_icon = ...

	####################
	# - Sockets
	####################
	input_sockets = {}
	input_socket_sets = {
		'freq_to_vacwl': {
			'freq': sockets.PhysicalFreqSocketDef(
				label='Freq',
			),
		},
		'vacwl_to_freq': {
			'vacwl': sockets.PhysicalVacWLSocketDef(
				label='Vac WL',
			),
		},
	}
	output_sockets = {}
	output_socket_sets = {
		'freq_to_vacwl': {
			'vacwl': sockets.PhysicalVacWLSocketDef(
				label='Vac WL',
			),
		},
		'vacwl_to_freq': {
			'freq': sockets.PhysicalFreqSocketDef(
				label='Freq',
			),
		},
	}

	####################
	# - Output Socket Computation
	####################
	@base.computes_output_socket('freq')
	def compute_freq(self: contracts.NodeTypeProtocol) -> sp.Expr:
		vac_speed_of_light = (
			sc.constants.speed_of_light * spu.meter / spu.second
		)

		vacwl = self.compute_input('vacwl')

		return spu.convert_to(
			vac_speed_of_light / vacwl,
			spu.hertz,
		)

	@base.computes_output_socket('vacwl')
	def compute_vacwl(self: contracts.NodeTypeProtocol) -> sp.Expr:
		vac_speed_of_light = (
			sc.constants.speed_of_light * spu.meter / spu.second
		)

		freq = self.compute_input('freq')

		return spu.convert_to(
			vac_speed_of_light / freq,
			spu.meter,
		)


####################
# - Blender Registration
####################
BL_REGISTER = [
	WaveConverterNode,
]
BL_NODES = {
	contracts.NodeType.WaveConverter: (
		contracts.NodeCategory.MAXWELLSIM_UTILITIES_CONVERTERS
	)
}
