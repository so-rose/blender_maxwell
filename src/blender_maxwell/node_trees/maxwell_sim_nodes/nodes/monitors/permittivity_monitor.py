# blender_maxwell
# Copyright (C) 2024 blender_maxwell Project Contributors
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import typing as typ

import sympy as sp
import tidy3d as td

from blender_maxwell.assets.geonodes import GeoNodes, import_geonodes
from blender_maxwell.utils import logger
from blender_maxwell.utils import sympy_extra as spux

from ... import contracts as ct
from ... import managed_objs, sockets
from .. import base, events

log = logger.get(__name__)

FK = ct.FlowKind
FS = ct.FlowSignal
MT = spux.MathType
PT = spux.PhysicalType


class PermittivityMonitorNode(base.MaxwellSimNode):
	"""Provides a bounded 1D/2D/3D recording region for the diagonal of the complex-valued permittivity tensor."""

	node_type = ct.NodeType.PermittivityMonitor
	bl_label = 'Permittivity Monitor'
	use_sim_node_name = True

	####################
	# - Sockets
	####################
	input_sockets: typ.ClassVar = {
		'Center': sockets.ExprSocketDef(
			size=spux.NumberSize1D.Vec3,
			physical_type=PT.Length,
		),
		'Size': sockets.ExprSocketDef(
			size=spux.NumberSize1D.Vec3,
			physical_type=PT.Length,
			default_value=sp.ImmutableMatrix([1, 1, 1]),
			abs_min=0,
			abs_min_closed=False,
		),
		'Stride': sockets.ExprSocketDef(
			size=spux.NumberSize1D.Vec3,
			mathtype=MT.Integer,
			default_value=sp.ImmutableMatrix([10, 10, 10]),
			abs_min=1,
		),
		'Freqs': sockets.ExprSocketDef(
			active_kind=FK.Range,
			physical_type=PT.Freq,
			default_unit=spux.THz,
			default_min=374.7406,  ## 800nm
			default_max=1498.962,  ## 200nm
			default_steps=100,
		),
	}
	output_sockets: typ.ClassVar = {
		'Monitor': sockets.MaxwellMonitorSocketDef(),
	}

	managed_obj_types: typ.ClassVar = {
		'modifier': managed_objs.ManagedBLModifier,
	}

	####################
	# - FlowKind.Value
	####################
	@events.computes_output_socket(
		'Monitor',
		kind=FK.Value,
		# Loaded
		outscks_kinds={
			'Monitor': {FK.Func, FK.Params},
		},
	)
	def compute_value(self, output_sockets) -> ct.ParamsFlow | FS:
		"""Realizes the output function w/output parameters."""
		value = events.realize_known(output_sockets['Monitor'])
		if value is not None:
			return value
		return FS.FlowPending

	####################
	# - FlowKind.Func
	####################
	@events.computes_output_socket(
		'Monitor',
		kind=FK.Func,
		# Loaded
		props={'sim_node_name'},
		inscks_kinds={
			'Center': FK.Func,
			'Size': FK.Func,
			'Stride': FK.Func,
			'Freqs': FK.Func,
		},
		scale_input_sockets={
			'Center': ct.UNITS_TIDY3D,
			'Size': ct.UNITS_TIDY3D,
			'Freqs': ct.UNITS_TIDY3D,
		},
	)
	def compute_func(self, props, input_sockets) -> td.FieldMonitor:
		"""Lazily assemble the permittivity monitor from the input functions."""
		center = input_sockets['Center']
		size = input_sockets['Size']
		stride = input_sockets['Stride']
		freqs = input_sockets['Freqs']

		sim_node_name = props['sim_node_name']

		return (center | size | stride | freqs).compose_within(
			lambda els: td.PermittivityMonitor(
				name=sim_node_name,
				center=els[0].flatten().tolist(),
				size=els[1].flatten().tolist(),
				interval_space=els[2].flatten().tolist(),
				freqs=els[3].flatten(),
			)
		)

	####################
	# - FlowKind.Params
	####################
	@events.computes_output_socket(
		'Monitor',
		kind=FK.Params,
		# Loaded
		inscks_kinds={
			'Center': FK.Params,
			'Size': FK.Params,
			'Stride': FK.Params,
			'Freqs': FK.Params,
		},
	)
	def compute_params(self, input_sockets) -> td.FieldMonitor:
		center = input_sockets['Center']
		size = input_sockets['Size']
		stride = input_sockets['Stride']
		freqs = input_sockets['Freqs']

		return center | size | stride | freqs

	####################
	# - Preview
	####################
	@events.computes_output_socket(
		'Permittivity Monitor',
		kind=FK.Previews,
		# Loaded
		props={'sim_node_name'},
	)
	def compute_previews_freq(self, props):
		"""Mark the monitor as participating in the preview."""
		return ct.PreviewsFlow(bl_object_names={props['sim_node_name']})

	@events.on_value_changed(
		# Trigger
		socket_name={'Center', 'Size'},
		run_on_init=True,
		# Loaded
		managed_objs={'modifier'},
		inscks_kinds={
			'Center': {FK.Func, FK.Params},
			'Size': {FK.Func, FK.Params},
		},
		scale_input_sockets={
			'Center': ct.UNITS_BLENDER,
			'Size': ct.UNITS_BLENDER,
		},
	)
	def on_previewable_changed(self, managed_objs, input_sockets):
		"""Push changes in the inputs to the center / size."""
		center = events.realize_preview(input_sockets['Center'])
		size = events.realize_preview(input_sockets['Size'])

		# Push Input Values to GeoNodes Modifier
		managed_objs['modifier'].bl_modifier(
			'NODES',
			{
				'node_group': import_geonodes(GeoNodes.MonitorPermittivity),
				'inputs': {
					'Size': size,
				},
			},
			location=center,
		)


####################
# - Blender Registration
####################
BL_REGISTER = [
	PermittivityMonitorNode,
]
BL_NODES = {ct.NodeType.PermittivityMonitor: (ct.NodeCategory.MAXWELLSIM_MONITORS)}
