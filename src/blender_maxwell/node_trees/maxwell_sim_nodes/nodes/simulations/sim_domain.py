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
from blender_maxwell.utils import sympy_extra as spux
from blender_maxwell.utils import logger

from ... import contracts as ct
from ... import managed_objs, sockets
from .. import base, events

log = logger.get(__name__)


class SimDomainNode(base.MaxwellSimNode):
	"""The domain of a simulation in space and time, including bounds, discretization strategy, and the ambient medium."""

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
	# - FlowKind.Value
	####################
	@events.computes_output_socket(
		'Domain',
		kind=ct.FlowKind.Value,
		# Loaded
		output_sockets={'Domain'},
		output_socket_kinds={'Domain': {ct.FlowKind.Func, ct.FlowKind.Params}},
	)
	def compute_domain_value(self, output_sockets) -> ct.ParamsFlow | ct.FlowSignal:
		"""Compute the particular value of the simulation domain from strictly non-symbolic inputs."""
		output_func = output_sockets['Domain'][ct.FlowKind.Func]
		output_params = output_sockets['Domain'][ct.FlowKind.Params]

		has_output_func = not ct.FlowSignal.check(output_func)
		has_output_params = not ct.FlowSignal.check(output_params)

		if has_output_func and has_output_params and not output_params.symbols:
			return output_func.realize(output_params)
		return ct.FlowSignal.FlowPending

	####################
	# - FlowKind.Func
	####################
	@events.computes_output_socket(
		'Domain',
		kind=ct.FlowKind.Func,
		# Loaded
		input_sockets={'Duration', 'Center', 'Size', 'Grid', 'Ambient Medium'},
		input_socket_kinds={
			'Duration': ct.FlowKind.Func,
			'Center': ct.FlowKind.Func,
			'Size': ct.FlowKind.Func,
			'Grid': ct.FlowKind.Func,
			'Ambient Medium': ct.FlowKind.Func,
		},
	)
	def compute_domain_func(self, input_sockets) -> ct.ParamsFlow | ct.FlowSignal:
		"""Compute the particular value of the simulation domain from strictly non-symbolic inputs."""
		duration = input_sockets['Duration']
		center = input_sockets['Center']
		size = input_sockets['Size']
		grid = input_sockets['Grid']
		medium = input_sockets['Ambient Medium']

		has_duration = not ct.FlowSignal.check(duration)
		has_center = not ct.FlowSignal.check(center)
		has_size = not ct.FlowSignal.check(size)
		has_grid = not ct.FlowSignal.check(grid)
		has_medium = not ct.FlowSignal.check(medium)

		if has_duration and has_center and has_size and has_grid and has_medium:
			return (
				duration.scale_to_unit_system(ct.UNITS_TIDY3D)
				| center.scale_to_unit_system(ct.UNITS_TIDY3D)
				| size.scale_to_unit_system(ct.UNITS_TIDY3D)
				| grid
				| medium
			).compose_within(
				lambda els: {
					'run_time': els[0],
					'center': els[1].flatten().tolist(),
					'size': els[2].flatten().tolist(),
					'grid_spec': els[3],
					'medium': els[4],
				},
				supports_jax=False,
			)
		return ct.FlowSignal.FlowPending

	####################
	# - FlowKind.Params
	####################
	@events.computes_output_socket(
		'Domain',
		kind=ct.FlowKind.Params,
		# Loaded
		input_sockets={'Duration', 'Center', 'Size', 'Grid', 'Ambient Medium'},
		input_socket_kinds={
			'Duration': ct.FlowKind.Params,
			'Center': ct.FlowKind.Params,
			'Size': ct.FlowKind.Params,
			'Grid': ct.FlowKind.Params,
			'Ambient Medium': ct.FlowKind.Params,
		},
	)
	def compute_domain_params(self, input_sockets) -> ct.ParamsFlow | ct.FlowSignal:
		"""Compute the output `ParamsFlow` of the simulation domain from strictly non-symbolic inputs."""
		duration = input_sockets['Duration']
		center = input_sockets['Center']
		size = input_sockets['Size']
		grid = input_sockets['Grid']
		medium = input_sockets['Ambient Medium']

		has_duration = not ct.FlowSignal.check(duration)
		has_center = not ct.FlowSignal.check(center)
		has_size = not ct.FlowSignal.check(size)
		has_grid = not ct.FlowSignal.check(grid)
		has_medium = not ct.FlowSignal.check(medium)

		if has_duration and has_center and has_size and has_grid and has_medium:
			return duration | center | size | grid | medium
		return ct.FlowSignal.FlowPending

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
		"""Mark the managed preview object for preview when `Domain` is linked to a viewer."""
		return ct.PreviewsFlow(bl_object_names={props['sim_node_name']})

	@events.on_value_changed(
		# Trigger
		socket_name={'Center', 'Size'},
		run_on_init=True,
		# Loaded
		input_sockets={'Center', 'Size'},
		managed_objs={'modifier'},
		output_sockets={'Domain'},
		output_socket_kinds={'Domain': ct.FlowKind.Params},
	)
	def on_input_changed(self, managed_objs, input_sockets, output_sockets) -> None:
		"""Preview the simulation domain based on input parameters, so long as they are not dependent on unrealized symbols."""
		output_params = output_sockets['Domain']
		center = input_sockets['Center']

		has_output_params = not ct.FlowSignal.check(output_params)
		has_center = not ct.FlowSignal.check(center)

		if has_center and has_output_params and not output_params.symbols:
			# Push Loose Input Values to GeoNodes Modifier
			managed_objs['modifier'].bl_modifier(
				'NODES',
				{
					'node_group': import_geonodes(GeoNodes.SimulationSimDomain),
					'unit_system': ct.UNITS_BLENDER,
					'inputs': {
						'Size': input_sockets['Size'],
					},
				},
				location=spux.scale_to_unit_system(center, ct.UNITS_BLENDER),
			)


####################
# - Blender Registration
####################
BL_REGISTER = [
	SimDomainNode,
]
BL_NODES = {ct.NodeType.SimDomain: (ct.NodeCategory.MAXWELLSIM_SIMS)}
