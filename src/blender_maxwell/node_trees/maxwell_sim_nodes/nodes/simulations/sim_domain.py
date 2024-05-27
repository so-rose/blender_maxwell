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
import sympy.physics.units as spu

from blender_maxwell.assets.geonodes import GeoNodes, import_geonodes
from blender_maxwell.utils import extra_sympy_units as spux
from blender_maxwell.utils import logger

from ... import contracts as ct
from ... import managed_objs, sockets
from .. import base, events

log = logger.get(__name__)


class SimDomainNode(base.MaxwellSimNode):
	node_type = ct.NodeType.SimDomain
	bl_label = 'Sim Domain'
	use_sim_node_name = True

	input_sockets: typ.ClassVar = {
		'Duration': sockets.ExprSocketDef(
			physical_type=spux.PhysicalType.Time,
			default_unit=spu.picosecond,
			default_value=5,
			abs_min=0,
		),
		'Center': sockets.ExprSocketDef(
			size=spux.NumberSize1D.Vec3,
			mathtype=spux.MathType.Real,
			physical_type=spux.PhysicalType.Length,
			default_unit=spu.micrometer,
			default_value=sp.Matrix([0, 0, 0]),
		),
		'Size': sockets.ExprSocketDef(
			size=spux.NumberSize1D.Vec3,
			mathtype=spux.MathType.Real,
			physical_type=spux.PhysicalType.Length,
			default_unit=spu.micrometer,
			default_value=sp.Matrix([1, 1, 1]),
			abs_min=0.001,
		),
		'Grid': sockets.MaxwellSimGridSocketDef(),
		'Ambient Medium': sockets.MaxwellMediumSocketDef(),
	}
	output_sockets: typ.ClassVar = {
		'Domain': sockets.MaxwellSimDomainSocketDef(),
	}

	managed_obj_types: typ.ClassVar = {
		'modifier': managed_objs.ManagedBLModifier,
	}

	####################
	# - Outputs
	####################
	@events.computes_output_socket(
		'Domain',
		input_sockets={'Duration', 'Center', 'Size', 'Grid', 'Ambient Medium'},
		unit_systems={'Tidy3DUnits': ct.UNITS_TIDY3D},
		scale_input_sockets={
			'Duration': 'Tidy3DUnits',
			'Center': 'Tidy3DUnits',
			'Size': 'Tidy3DUnits',
		},
	)
	def compute_domain(self, input_sockets, unit_systems) -> sp.Expr:
		return {
			'run_time': input_sockets['Duration'],
			'center': input_sockets['Center'],
			'size': input_sockets['Size'],
			'grid_spec': input_sockets['Grid'],
			'medium': input_sockets['Ambient Medium'],
		}

	####################
	# - Preview
	####################
	@events.computes_output_socket(
		'Domain',
		kind=ct.FlowKind.Previews,
		# Loaded
		props={'sim_node_name'},
	)
	def compute_previews(self, props):
		return ct.PreviewsFlow(bl_object_names={props['sim_node_name']})

	@events.on_value_changed(
		## Trigger
		socket_name={'Center', 'Size'},
		run_on_init=True,
		# Loaded
		input_sockets={'Center', 'Size'},
		managed_objs={'modifier'},
		unit_systems={'BlenderUnits': ct.UNITS_BLENDER},
		scale_input_sockets={
			'Center': 'BlenderUnits',
		},
	)
	def on_input_changed(
		self,
		managed_objs,
		input_sockets,
		unit_systems,
	):
		# Push Loose Input Values to GeoNodes Modifier
		managed_objs['modifier'].bl_modifier(
			'NODES',
			{
				'node_group': import_geonodes(GeoNodes.SimulationSimDomain),
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
	SimDomainNode,
]
BL_NODES = {ct.NodeType.SimDomain: (ct.NodeCategory.MAXWELLSIM_SIMS)}
