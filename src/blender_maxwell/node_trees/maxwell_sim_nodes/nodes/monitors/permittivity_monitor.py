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
from blender_maxwell.utils import sympy_extra as spux
from blender_maxwell.utils import logger

from ... import contracts as ct
from ... import managed_objs, sockets
from .. import base, events

log = logger.get(__name__)


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
			physical_type=spux.PhysicalType.Length,
		),
		'Size': sockets.ExprSocketDef(
			size=spux.NumberSize1D.Vec3,
			physical_type=spux.PhysicalType.Length,
			default_value=sp.Matrix([1, 1, 1]),
			abs_min=0,
		),
		'Stride': sockets.ExprSocketDef(
			size=spux.NumberSize1D.Vec3,
			mathtype=spux.MathType.Integer,
			default_value=sp.Matrix([10, 10, 10]),
			abs_min=0,
		),
		'Freqs': sockets.ExprSocketDef(
			active_kind=ct.FlowKind.Range,
			physical_type=spux.PhysicalType.Freq,
			default_unit=spux.THz,
			default_min=374.7406,  ## 800nm
			default_max=1498.962,  ## 200nm
			default_steps=100,
		),
	}
	output_sockets: typ.ClassVar = {
		'Permittivity Monitor': sockets.MaxwellMonitorSocketDef()
	}

	managed_obj_types: typ.ClassVar = {
		'modifier': managed_objs.ManagedBLModifier,
	}

	####################
	# - Output
	####################
	@events.computes_output_socket(
		'Permittivity Monitor',
		props={'sim_node_name'},
		input_sockets={
			'Center',
			'Size',
			'Stride',
			'Freqs',
		},
		input_socket_kinds={
			'Freqs': ct.FlowKind.Range,
		},
		unit_systems={'Tidy3DUnits': ct.UNITS_TIDY3D},
		scale_input_sockets={
			'Center': 'Tidy3DUnits',
			'Size': 'Tidy3DUnits',
			'Freqs': 'Tidy3DUnits',
		},
	)
	def compute_permittivity_monitor(
		self,
		input_sockets: dict,
		props: dict,
		unit_systems: dict,
	) -> td.FieldMonitor:
		log.info(
			'Computing PermittivityMonitor (name="%s") with center="%s", size="%s"',
			props['sim_node_name'],
			input_sockets['Center'],
			input_sockets['Size'],
		)
		return td.PermittivityMonitor(
			center=input_sockets['Center'],
			size=input_sockets['Size'],
			name=props['sim_node_name'],
			interval_space=tuple(input_sockets['Stride']),
			freqs=input_sockets['Freqs'].realize().values,
		)

	####################
	# - Preview
	####################
	@events.computes_output_socket(
		'Permittivity Monitor',
		kind=ct.FlowKind.Previews,
		# Loaded
		props={'sim_node_name'},
	)
	def compute_previews_freq(self, props):
		return ct.PreviewsFlow(bl_object_names={props['sim_node_name']})

	@events.on_value_changed(
		# Trigger
		socket_name={'Center', 'Size'},
		run_on_init=True,
		# Loaded
		managed_objs={'modifier'},
		input_sockets={'Center', 'Size'},
		unit_systems={'BlenderUnits': ct.UNITS_BLENDER},
		scale_input_sockets={
			'Center': 'BlenderUnits',
		},
	)
	def on_inputs_changed(
		self,
		managed_objs: dict,
		input_sockets: dict,
		unit_systems: dict,
	):
		# Push Input Values to GeoNodes Modifier
		managed_objs['modifier'].bl_modifier(
			'NODES',
			{
				'node_group': import_geonodes(GeoNodes.MonitorPermittivity),
				'unit_system': unit_systems['BlenderUnits'],
				'inputs': {
					'Size': input_sockets['Size'],
				},
			},
			location=input_sockets['Center'],
		)


####################
# - Blender Registration
####################
BL_REGISTER = [
	PermittivityMonitorNode,
]
BL_NODES = {ct.NodeType.PermittivityMonitor: (ct.NodeCategory.MAXWELLSIM_MONITORS)}
