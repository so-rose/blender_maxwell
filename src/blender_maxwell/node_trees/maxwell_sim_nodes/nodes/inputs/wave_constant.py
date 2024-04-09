import typing as typ

import bpy
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
		'Wavelength': {},
		'Frequency': {},
	}

	use_range: bpy.props.BoolProperty(
		name='Range',
		description='Whether to use the wavelength range',
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
		all_loose_input_sockets=True,
	)
	def compute_wl(self, loose_input_sockets: dict) -> sp.Expr:
		if (wl := loose_input_sockets.get('WL')) is not None:
			return wl

		freq = loose_input_sockets.get('Freq')

		if isinstance(freq, ct.LazyDataValueRange):
			return freq.rescale_bounds(
				lambda bound: constants.vac_speed_of_light / bound, reverse=True
			)

		return constants.vac_speed_of_light / freq

	@events.computes_output_socket(
		'Freq',
		all_loose_input_sockets=True,
	)
	def compute_freq(self, loose_input_sockets: dict) -> sp.Expr:
		if (freq := loose_input_sockets.get('Freq')) is not None:
			return freq

		wl = loose_input_sockets.get('WL')

		if isinstance(wl, ct.LazyDataValueRange):
			return wl.rescale_bounds(
				lambda bound: constants.vac_speed_of_light / bound, reverse=True
			)

		return constants.vac_speed_of_light / wl

	####################
	# - Event Methods
	####################
	@events.on_value_changed(
		prop_name={'active_socket_set', 'use_range'},
		props={'active_socket_set', 'use_range'},
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

	@events.on_init(
		props={'active_socket_set', 'use_range'},
	)
	def on_init(self, props: dict):
		self.on_input_spec_change()


####################
# - Blender Registration
####################
BL_REGISTER = [
	WaveConstantNode,
]
BL_NODES = {ct.NodeType.WaveConstant: (ct.NodeCategory.MAXWELLSIM_INPUTS)}
