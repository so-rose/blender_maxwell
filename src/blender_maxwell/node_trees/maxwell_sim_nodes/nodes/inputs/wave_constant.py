import typing as typ

import bpy
import sympy as sp
import sympy.physics.units as spu

from blender_maxwell.utils import extra_sympy_units as spux
from blender_maxwell.utils import logger
from blender_maxwell.utils import sci_constants as constants

from ... import contracts as ct
from ... import sockets
from .. import base, events

log = logger.get(__name__)


class WaveConstantNode(base.MaxwellSimNode):
	node_type = ct.NodeType.WaveConstant
	bl_label = 'Wave Constant'

	input_socket_sets: typ.ClassVar = {
		'Wavelength': {},
		'Frequency': {},
	}

	use_range: bpy.props.BoolProperty(
		name='Range',
		description='Whether to use a wavelength/frequency range',
		default=False,
		update=lambda self, context: self.sync_prop('use_range', context),
	)

	def draw_props(self, _: bpy.types.Context, col: bpy.types.UILayout):
		col.prop(self, 'use_range', toggle=True)

	####################
	# - Event Methods: Wavelength Output
	####################
	@events.computes_output_socket(
		'WL',
		kind=ct.FlowKind.Value,
		# Data
		input_sockets={'WL', 'Freq'},
		input_sockets_optional={'WL': True, 'Freq': True},
	)
	def compute_wl_value(self, input_sockets: dict) -> sp.Expr:
		if input_sockets['WL'] is not None:
			return input_sockets['WL']

		if input_sockets['WL'] is None and input_sockets['Freq'] is None:
			msg = 'Both WL and Freq are None.'
			raise RuntimeError(msg)

		return constants.vac_speed_of_light / input_sockets['Freq']

	@events.computes_output_socket(
		'Freq',
		kind=ct.FlowKind.Value,
		# Data
		input_sockets={'WL', 'Freq'},
		input_sockets_optional={'WL': True, 'Freq': True},
	)
	def compute_freq_value(self, input_sockets: dict) -> sp.Expr:
		log.critical(input_sockets)
		if input_sockets['Freq'] is not None:
			return input_sockets['Freq']

		if input_sockets['WL'] is None and input_sockets['Freq'] is None:
			msg = 'Both WL and Freq are None.'
			raise RuntimeError(msg)

		return constants.vac_speed_of_light / input_sockets['WL']

	@events.computes_output_socket(
		'WL',
		kind=ct.FlowKind.LazyValueRange,
		# Data
		input_sockets={'WL', 'Freq'},
		input_sockets_optional={'WL': True, 'Freq': True},
	)
	def compute_wl_range(self, input_sockets: dict) -> sp.Expr:
		if input_sockets['WL'] is not None:
			return input_sockets['WL']

		if input_sockets['WL'] is None and input_sockets['Freq'] is None:
			msg = 'Both WL and Freq are None.'
			raise RuntimeError(msg)

		return input_sockets['Freq'].rescale_bounds(
			lambda bound: constants.vac_speed_of_light / bound, reverse=True
		)

	@events.computes_output_socket(
		'Freq',
		kind=ct.FlowKind.LazyValueRange,
		# Data
		input_sockets={'WL', 'Freq'},
		input_socket_kinds={
			'WL': ct.FlowKind.LazyValueRange,
			'Freq': ct.FlowKind.LazyValueRange,
		},
		input_sockets_optional={'WL': True, 'Freq': True},
	)
	def compute_freq_range(self, input_sockets: dict) -> sp.Expr:
		if input_sockets['Freq'] is not None:
			return input_sockets['Freq']

		if input_sockets['WL'] is None and input_sockets['Freq'] is None:
			msg = 'Both WL and Freq are None.'
			raise RuntimeError(msg)

		return input_sockets['WL'].rescale_bounds(
			lambda bound: constants.vac_speed_of_light / bound, reverse=True
		)

	####################
	# - Event Methods
	####################
	@events.on_value_changed(
		prop_name={'active_socket_set', 'use_range'},
		props={'active_socket_set', 'use_range'},
		run_on_init=True,
	)
	def on_input_spec_change(self, props: dict):
		if props['active_socket_set'] == 'Wavelength':
			self.loose_input_sockets = {
				'WL': sockets.PhysicalLengthSocketDef(
					is_array=props['use_range'],
					default_value=500 * spu.nm,
					default_unit=spu.nm,
				)
			}
		else:
			self.loose_input_sockets = {
				'Freq': sockets.PhysicalFreqSocketDef(
					is_array=props['use_range'],
					default_value=600 * spux.THz,
					default_unit=spux.THz,
				)
			}

		self.loose_output_sockets = {
			'WL': sockets.PhysicalLengthSocketDef(is_array=props['use_range']),
			'Freq': sockets.PhysicalFreqSocketDef(is_array=props['use_range']),
		}


####################
# - Blender Registration
####################
BL_REGISTER = [
	WaveConstantNode,
]
BL_NODES = {ct.NodeType.WaveConstant: (ct.NodeCategory.MAXWELLSIM_INPUTS)}
