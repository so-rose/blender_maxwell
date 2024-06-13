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

"""Implements `SimDomainNode`."""

import typing as typ

import sympy as sp
import sympy.physics.units as spu

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


class SimDomainNode(base.MaxwellSimNode):
	"""The domain of a simulation in space and time, including bounds, discretization strategy, and the ambient medium."""

	node_type = ct.NodeType.SimDomain
	bl_label = 'Sim Domain'
	use_sim_node_name = True

	input_sockets: typ.ClassVar = {
		'Duration': sockets.ExprSocketDef(
			physical_type=PT.Time,
			default_unit=spu.picosecond,
			default_value=5,
			abs_min=0,
		),
		'Center': sockets.ExprSocketDef(
			size=spux.NumberSize1D.Vec3,
			mathtype=MT.Real,
			physical_type=PT.Length,
			default_unit=spu.micrometer,
			default_value=sp.ImmutableMatrix([0, 0, 0]),
		),
		'Size': sockets.ExprSocketDef(
			size=spux.NumberSize1D.Vec3,
			mathtype=MT.Real,
			physical_type=PT.Length,
			default_unit=spu.micrometer,
			default_value=sp.ImmutableMatrix([1, 1, 1]),
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
	# - FlowKind.Value
	####################
	@events.computes_output_socket(
		'Domain',
		kind=FK.Value,
		# Loaded
		outscks_kinds={
			'Domain': {FK.Func, FK.Params},
		},
	)
	def compute_domain_value(self, output_sockets) -> ct.ParamsFlow | FS:
		"""Compute the particular value of the simulation domain from strictly non-symbolic inputs."""
		value = events.realize_known(output_sockets['Domain'])
		if value is not None:
			return value
		return FS.FlowPending

	####################
	# - FlowKind.Func
	####################
	@events.computes_output_socket(
		'Domain',
		kind=FK.Func,
		# Loaded
		input_sockets={'Duration', 'Center', 'Size', 'Grid', 'Ambient Medium'},
		input_socket_kinds={
			'Duration': FK.Func,
			'Center': FK.Func,
			'Size': FK.Func,
			'Grid': FK.Func,
			'Ambient Medium': FK.Func,
		},
		scale_input_sockets={
			'Duration': ct.UNITS_TIDY3D,
			'Center': ct.UNITS_TIDY3D,
			'Size': ct.UNITS_TIDY3D,
		},
	)
	def compute_domain_func(self, input_sockets) -> ct.ParamsFlow | FS:
		"""Compute the particular value of the simulation domain from strictly non-symbolic inputs."""
		duration = input_sockets['Duration']
		center = input_sockets['Center']
		size = input_sockets['Size']
		grid = input_sockets['Grid']
		medium = input_sockets['Ambient Medium']
		return (duration | center | size | grid | medium).compose_within(
			lambda els: {
				'run_time': els[0],
				'center': els[1].flatten().tolist(),
				'size': els[2].flatten().tolist(),
				'grid_spec': els[3],
				'medium': els[4],
			},
			supports_jax=False,
		)

	####################
	# - FlowKind.Params
	####################
	@events.computes_output_socket(
		'Domain',
		kind=FK.Params,
		# Loaded
		input_sockets={'Duration', 'Center', 'Size', 'Grid', 'Ambient Medium'},
		input_socket_kinds={
			'Duration': FK.Params,
			'Center': FK.Params,
			'Size': FK.Params,
			'Grid': FK.Params,
			'Ambient Medium': FK.Params,
		},
	)
	def compute_domain_params(self, input_sockets) -> ct.ParamsFlow | FS:
		"""Compute the output `ParamsFlow` of the simulation domain from strictly non-symbolic inputs."""
		duration = input_sockets['Duration']
		center = input_sockets['Center']
		size = input_sockets['Size']
		grid = input_sockets['Grid']
		medium = input_sockets['Ambient Medium']

		return duration | center | size | grid | medium

	####################
	# - Preview
	####################
	@events.computes_output_socket(
		'Domain',
		kind=FK.Previews,
		# Loaded
		props={'sim_node_name'},
	)
	def compute_previews(self, props):
		"""Mark the managed preview object for preview when `Domain` is linked to a viewer."""
		return ct.PreviewsFlow(bl_object_names={props['sim_node_name']})

	@events.on_value_changed(
		# Trigger
		socket_name={
			'Center': {FK.Func, FK.Params},
			'Size': {FK.Func, FK.Params},
		},
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
	def on_input_changed(self, managed_objs, input_sockets) -> None:
		"""Preview the simulation domain based on input parameters, so long as they are not dependent on unrealized symbols."""
		center = events.realize_preview(input_sockets['Center'])
		size = events.realize_preview(input_sockets['Size'])

		managed_objs['modifier'].bl_modifier(
			'NODES',
			{
				'node_group': import_geonodes(GeoNodes.SimulationSimDomain),
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
	SimDomainNode,
]
BL_NODES = {ct.NodeType.SimDomain: (ct.NodeCategory.MAXWELLSIM_SIMS)}
