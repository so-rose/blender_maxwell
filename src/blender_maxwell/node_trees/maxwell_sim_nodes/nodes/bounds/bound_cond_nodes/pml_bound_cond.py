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

"""Implements `PMLBoundCondNode`."""

import typing as typ

import bpy
import sympy as sp
import tidy3d as td

from blender_maxwell.utils import sympy_extra as spux
from blender_maxwell.utils import logger

from .... import contracts as ct
from .... import sockets
from ... import base, events

log = logger.get(__name__)


class PMLBoundCondNode(base.MaxwellSimNode):
	r"""A "Perfectly Matched Layer" boundary condition, which is a theoretical medium that attempts to _perfectly_ absorb all outgoing waves, so as to represent "infinite space" in FDTD simulations.

	PML boundary conditions do so by inducing a **frequency-dependent attenuation** on all waves that are outside of the boundary, over the course of several layers.

	It is critical to note that a PML boundary can only absorb **propagating** waves.
	_Evanescent_ waves oscillating w/o any power flux, ex. close to structures, may actually be **amplified** by a PML boundary.
	This is the reasoning behind the $\frac{\lambda}{2}$-distance rule of thumb.

	For more theoretical details, please refer to the `tidy3d` resource: <https://docs.flexcompute.com/projects/tidy3d/en/latest/api/_autosummary/tidy3d.PML.html>

	Notes:
		**Ensure** that all simulation structures are $\approx \frac{\lambda}{2}$ from any PML boundary.

		This helps avoid the amplification of stray evanescent waves.

	Socket Sets:
		Simple: Only specify the number of PML layers.
			$12$ should cover the most common cases; $40$ should be extremely stable.
		Full: Specify the conductivity min/max that make up the PML, as well as the order of approximating polynomials.
			The meaning of the parameters are rooted in the mathematics that underlie the PML function - if that doesn't mean anything to you, then you should probably leave it alone!
			Since the value units are sim-relative, we've opted to show the scaling information in the node's UI, instead of coercing the values into any particular unit.
	"""

	node_type = ct.NodeType.PMLBoundCond
	bl_label = 'PML Bound Cond'

	####################
	# - Sockets
	####################
	input_sockets: typ.ClassVar = {
		'Layers': sockets.ExprSocketDef(
			size=spux.NumberSize1D.Scalar,
			mathtype=spux.MathType.Integer,
			abs_min=1,
			default_value=12,
		),
	}
	input_socket_sets: typ.ClassVar = {
		'Simple': {},
		'Full': {
			'σ Order': sockets.ExprSocketDef(
				size=spux.NumberSize1D.Scalar,
				mathtype=spux.MathType.Integer,
				abs_min=1,
				default_value=3,
			),
			'σ Range': sockets.ExprSocketDef(
				size=spux.NumberSize1D.Vec2,
				mathtype=spux.MathType.Real,
				default_value=sp.Matrix([0, 1.5]),
				abs_min=0,
			),
			'κ Order': sockets.ExprSocketDef(
				size=spux.NumberSize1D.Scalar,
				mathtype=spux.MathType.Integer,
				abs_min=1,
				default_value=3,
			),
			'κ Range': sockets.ExprSocketDef(
				size=spux.NumberSize1D.Vec2,
				mathtype=spux.MathType.Real,
				default_value=sp.Matrix([0, 1.5]),
				abs_min=0,
			),
			'α Order': sockets.ExprSocketDef(
				size=spux.NumberSize1D.Scalar,
				mathtype=spux.MathType.Integer,
				abs_min=1,
				default_value=3,
			),
			'α Range': sockets.ExprSocketDef(
				size=spux.NumberSize1D.Vec2,
				mathtype=spux.MathType.Real,
				default_value=sp.Matrix([0, 1.5]),
				abs_min=0,
			),
		},
	}
	output_sockets: typ.ClassVar = {
		'BC': sockets.MaxwellBoundCondSocketDef(),
	}

	####################
	# - UI
	####################
	def draw_info(self, _: bpy.types.Context, layout: bpy.types.UILayout) -> None:
		if self.active_socket_set == 'Full':
			box = layout.box()
			row = box.row()
			row.alignment = 'CENTER'
			row.label(text='Parameter Scale')

			# Split
			split = box.split(factor=0.4, align=False)

			## LHS: Parameter Names
			col = split.column()
			col.alignment = 'RIGHT'
			for param in ['σ', 'κ', 'α']:
				col.label(text=param + ':')

			## RHS: Parameter Units
			col = split.column()
			for _ in range(3):
				col.label(text='2ε₀/Δt')

	####################
	# - FlowKind.Value
	####################
	@events.computes_output_socket(
		'BC',
		kind=ct.FlowKind.Value,
		# Loaded
		props={'active_socket_set'},
		input_sockets={
			'Layers',
			'σ Order',
			'σ Range',
			'κ Order',
			'κ Range',
			'α Order',
			'α Range',
		},
		input_sockets_optional={
			'σ Order': True,
			'σ Range': True,
			'κ Order': True,
			'κ Range': True,
			'α Order': True,
			'α Range': True,
		},
		output_sockets={'BC'},
		output_socket_kinds={'BC': ct.FlowKind.Params},
	)
	def compute_pml_value(self, props, input_sockets, output_sockets) -> td.PML:
		r"""Computes the PML boundary condition based on the active socket set.

		- **Simple**: Use `tidy3d`'s default parameters for defining the PML conductor (apart from number of layers).
		- **Full**: Use the user-defined $\sigma$, $\kappa$, and $\alpha$ parameters, specifically polynomial order and sim-relative min/max conductivity values.
		"""
		output_params = output_sockets['BC']
		layers = input_sockets['Layers']

		has_layers = not ct.FlowSignal.check(layers)
		has_output_params = not ct.FlowSignal.check(output_params)

		if has_output_params and has_layers and not output_params.symbols:
			active_socket_set = props['active_socket_set']
			match active_socket_set:
				case 'Simple':
					return td.PML(num_layers=layers)

				case 'Full':
					sigma_order = input_sockets['σ Order']
					sigma_range = input_sockets['σ Range']
					kappa_order = input_sockets['κ Order']
					kappa_range = input_sockets['κ Range']
					alpha_order = input_sockets['α Order']
					alpha_range = input_sockets['α Range']

					has_sigma_order = not ct.FlowSignal.check(sigma_order)
					has_sigma_range = not ct.FlowSignal.check(sigma_range)
					has_kappa_order = not ct.FlowSignal.check(kappa_order)
					has_kappa_range = not ct.FlowSignal.check(kappa_range)
					has_alpha_order = not ct.FlowSignal.check(alpha_order)
					has_alpha_range = not ct.FlowSignal.check(alpha_range)

					if (
						has_sigma_order
						and has_sigma_range
						and has_kappa_order
						and has_kappa_range
						and has_alpha_order
						and has_alpha_range
					):
						return td.PML(
							num_layers=layers,
							parameters=td.PMLParams(
								sigma_order=sigma_order,
								sigma_min=sigma_range[0],
								sigma_max=sigma_range[1],
								kappa_order=kappa_order,
								kappa_min=kappa_range[0],
								kappa_max=kappa_range[1],
								alpha_order=alpha_order,
								alpha_min=alpha_range[0],
								alpha_max=alpha_range[1],
							),
						)

		return ct.FlowSignal.FlowPending

	####################
	# - FlowKind.Func
	####################
	@events.computes_output_socket(
		'BC',
		kind=ct.FlowKind.Func,
		# Loaded
		props={'active_socket_set'},
		input_sockets={
			'Layers',
			'σ Order',
			'σ Range',
			'κ Order',
			'κ Range',
			'α Order',
			'α Range',
		},
		input_socket_kinds={
			'Layers': ct.FlowKind.Func,
			'σ Order': ct.FlowKind.Func,
			'σ Range': ct.FlowKind.Func,
			'κ Order': ct.FlowKind.Func,
			'κ Range': ct.FlowKind.Func,
			'α Order': ct.FlowKind.Func,
			'α Range': ct.FlowKind.Func,
		},
		input_sockets_optional={
			'σ Order': True,
			'σ Range': True,
			'κ Order': True,
			'κ Range': True,
			'α Order': True,
			'α Range': True,
		},
		output_sockets={'BC'},
		output_socket_kinds={'BC': ct.FlowKind.Params},
	)
	def compute_pml_func(self, props, input_sockets, output_sockets) -> td.PML:
		output_params = output_sockets['BC']
		layers = input_sockets['Layers']

		has_output_params = not ct.FlowSignal.check(output_params)
		has_layers = not ct.FlowSignal.check(layers)

		if has_output_params and has_layers:
			active_socket_set = props['active_socket_set']
			match active_socket_set:
				case 'Simple':
					return layers.compose_within(
						enclosing_func=lambda layers: td.PML(num_layers=layers),
						supports_jax=False,
					)

				case 'Full':
					sigma_order = input_sockets['σ Order']
					sigma_range = input_sockets['σ Range']
					kappa_order = input_sockets['κ Order']
					kappa_range = input_sockets['κ Range']
					alpha_order = input_sockets['α Order']
					alpha_range = input_sockets['α Range']

					has_sigma_order = not ct.FlowSignal.check(sigma_order)
					has_sigma_range = not ct.FlowSignal.check(sigma_range)
					has_kappa_order = not ct.FlowSignal.check(kappa_order)
					has_kappa_range = not ct.FlowSignal.check(kappa_range)
					has_alpha_order = not ct.FlowSignal.check(alpha_order)
					has_alpha_range = not ct.FlowSignal.check(alpha_range)

					if (
						has_sigma_order
						and has_sigma_range
						and has_kappa_order
						and has_kappa_range
						and has_alpha_order
						and has_alpha_range
					):
						return (
							sigma_order
							| sigma_range
							| kappa_order
							| kappa_range
							| alpha_order
							| alpha_range
						).compose_within(
							enclosing_func=lambda els: td.PML(
								num_layers=layers,
								parameters=td.PMLParams(
									sigma_order=els[0],
									sigma_min=els[1][0],
									sigma_max=els[1][1],
									kappa_order=els[2],
									kappa_min=els[3][0],
									kappa_max=els[3][1],
									alpha_order=els[4][1],
									alpha_min=els[5][0],
									alpha_max=els[5][1],
								),
							)
						)

		return ct.FlowSignal.FlowPending

	####################
	# - FlowKind.Params
	####################
	@events.computes_output_socket(
		'BC',
		kind=ct.FlowKind.Params,
		# Loaded
		props={'active_socket_set'},
		input_sockets={
			'Layers',
			'σ Order',
			'σ Range',
			'κ Order',
			'κ Range',
			'α Order',
			'α Range',
		},
		input_socket_kinds={
			'Layers': ct.FlowKind.Params,
			'σ Order': ct.FlowKind.Params,
			'σ Range': ct.FlowKind.Params,
			'κ Order': ct.FlowKind.Params,
			'κ Range': ct.FlowKind.Params,
			'α Order': ct.FlowKind.Params,
			'α Range': ct.FlowKind.Params,
		},
		input_sockets_optional={
			'σ Order': True,
			'σ Range': True,
			'κ Order': True,
			'κ Range': True,
			'α Order': True,
			'α Range': True,
		},
	)
	def compute_pml_params(self, props, input_sockets) -> td.PML:
		r"""Computes the PML boundary condition based on the active socket set.

		- **Simple**: Use `tidy3d`'s default parameters for defining the PML conductor (apart from number of layers).
		- **Full**: Use the user-defined $\sigma$, $\kappa$, and $\alpha$ parameters, specifically polynomial order and sim-relative min/max conductivity values.
		"""
		layers = input_sockets['Layers']
		has_layers = not ct.FlowSignal.check(layers)

		if has_layers:
			active_socket_set = props['active_socket_set']
			match active_socket_set:
				case 'Simple':
					return layers

				case 'Full':
					sigma_order = input_sockets['σ Order']
					sigma_range = input_sockets['σ Range']
					kappa_order = input_sockets['σ Order']
					kappa_range = input_sockets['σ Range']
					alpha_order = input_sockets['σ Order']
					alpha_range = input_sockets['σ Range']

					has_sigma_order = not ct.FlowSignal.check(sigma_order)
					has_sigma_range = not ct.FlowSignal.check(sigma_range)
					has_kappa_order = not ct.FlowSignal.check(kappa_order)
					has_kappa_range = not ct.FlowSignal.check(kappa_range)
					has_alpha_order = not ct.FlowSignal.check(alpha_order)
					has_alpha_range = not ct.FlowSignal.check(alpha_range)

					if (
						has_sigma_order
						and has_sigma_range
						and has_kappa_order
						and has_kappa_range
						and has_alpha_order
						and has_alpha_range
					):
						return (
							sigma_order
							| sigma_range
							| kappa_order
							| kappa_range
							| alpha_order
							| alpha_range
						)

		return ct.FlowSignal.FlowPending


####################
# - Blender Registration
####################
BL_REGISTER = [
	PMLBoundCondNode,
]
BL_NODES = {ct.NodeType.PMLBoundCond: (ct.NodeCategory.MAXWELLSIM_BOUNDS)}
