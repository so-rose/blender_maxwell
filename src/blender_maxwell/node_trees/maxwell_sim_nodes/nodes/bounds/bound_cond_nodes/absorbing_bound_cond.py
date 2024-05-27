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

"""Implements `AdiabAbsorbBoundCondNode`."""

import typing as typ

import bpy
import sympy as sp
import tidy3d as td

from blender_maxwell.utils import extra_sympy_units as spux
from blender_maxwell.utils import logger

from .... import contracts as ct
from .... import sockets
from ... import base, events

log = logger.get(__name__)


class AdiabAbsorbBoundCondNode(base.MaxwellSimNode):
	r"""A boundary condition that generically (adiabatically) absorbs outgoing energy, by gradually ramping up the strength of the conductor over many layers, until a final PEC layer.

	Compared to PML, this boundary is more computationally expensive, and may result in higher reflectivity (and thus lower accuracy).
	The general reason to use it is **to fix divergence in cases where dispersive materials intersect the simulation boundary**.

	For more theoretical details, please refer to the `tidy3d` resource: <https://docs.flexcompute.com/projects/tidy3d/en/latest/api/_autosummary/tidy3d.Absorber.html>

	Notes:
		**Ensure** that all simulation structures are $\approx \frac{\lambda}{2}$ from any PML boundary.

		This helps avoid the amplification of stray evanescent waves.

	Socket Sets:
		Simple: Only specify the number of absorption layers.
			$40$ should generally be sufficient, but in the case of divergence issues, bumping up the number of layers should be the go-to remedy to try.
		Full: Specify the conductivity min/max that makes up the absorption up the PML, as well as the order of the polynomial used to scale the effect through the layers.
			You should probably leave this alone.
			Since the value units are sim-relative, we've opted to show the scaling information in the node's UI, instead of coercing the values into any particular unit.
	"""

	node_type = ct.NodeType.AdiabAbsorbBoundCond
	bl_label = 'Absorber Bound Cond'

	####################
	# - Sockets
	####################
	input_sockets: typ.ClassVar = {
		'Layers': sockets.ExprSocketDef(
			size=spux.NumberSize1D.Scalar,
			mathtype=spux.MathType.Integer,
			abs_min=1,
			default_value=40,
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
			col.label(text='σ:')

			## RHS: Parameter Units
			col = split.column()
			col.label(text='2ε₀/Δt')

	####################
	# - FlowKind.Value
	####################
	@events.computes_output_socket(
		'BC',
		# Loaded
		props={'active_socket_set'},
		input_sockets={
			'Layers',
			'σ Order',
			'σ Range',
		},
		input_sockets_optional={
			'σ Order': True,
			'σ Range': True,
		},
		output_sockets={'BC'},
		output_socket_kinds={'BC': ct.FlowKind.Params},
	)
	def compute_bc_value(self, props, input_sockets, output_sockets) -> td.Absorber:
		r"""Computes the adiabatic absorber boundary condition based on the active socket set.

		- **Simple**: Use `tidy3d`'s default parameters for defining the absorber parameters (apart from number of layers).
		- **Full**: Use the user-defined $\sigma$ parameters, specifically polynomial order and sim-relative min/max conductivity values.
		"""
		output_params = output_sockets['BC']
		layers = input_sockets['Layers']

		has_output_params = not ct.FlowSignal.check(output_params)
		has_layers = not ct.FlowSignal.check(layers)

		active_socket_set = props['active_socket_set']
		if has_layers and has_output_params and not output_params.symbols:
			# Simple PML
			if active_socket_set == 'Simple':
				return td.Absorber(num_layers=layers)

			# Full PML
			sig_order = input_sockets['σ Order']
			sig_range = input_sockets['σ Range']

			has_sig_order = not ct.FlowSignal.check(sig_order)
			has_sig_range = not ct.FlowSignal.check(sig_range)

			if has_sig_order and has_sig_range:
				return td.Absorber(
					num_layers=layers,
					parameters=td.AbsorberParams(
						sigma_order=sig_order,
						sigma_min=sig_range[0],
						sigma_max=sig_range[1],
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
		},
		input_socket_kinds={
			'Layers': ct.FlowKind.Func,
			'σ Order': ct.FlowKind.Func,
			'σ Range': ct.FlowKind.Func,
		},
		input_sockets_optional={
			'σ Order': True,
			'σ Range': True,
		},
		output_sockets={'BC'},
		output_socket_kinds={'BC': ct.FlowKind.Params},
	)
	def compute_bc_func(self, props, input_sockets, output_sockets) -> td.Absorber:
		r"""Computes the adiabatic absorber boundary condition based on the active socket set.

		- **Simple**: Use `tidy3d`'s default parameters for defining the absorber parameters (apart from number of layers).
		- **Full**: Use the user-defined $\sigma$ parameters, specifically polynomial order and sim-relative min/max conductivity values.
		"""
		layers = input_sockets['Layers']

		has_layers = not ct.FlowSignal.check(layers)

		active_socket_set = props['active_socket_set']
		if has_layers:
			# Simple PML
			if active_socket_set == 'Simple':
				return layers.compose_within(
					enclosing_func=lambda _layers: td.Absorber(num_layers=_layers),
					supports_jax=False,
				)

			# Full PML
			sig_order = input_sockets['σ Order']
			sig_range = input_sockets['σ Range']

			has_sig_order = not ct.FlowSignal.check(sig_order)
			has_sig_range = not ct.FlowSignal.check(sig_range)

			if has_sig_order and has_sig_range:
				return (layers | sig_order | sig_range).compose_within(
					enclosing_func=lambda els: td.Absorber(
						num_layers=els[0],
						parameters=td.AbsorberParams(
							sigma_order=els[1],
							sigma_min=els[2][0],
							sigma_max=els[2][1],
						),
					),
					supports_jax=False,
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
		},
		input_socket_kinds={
			'Layers': ct.FlowKind.Params,
			'σ Order': ct.FlowKind.Params,
			'σ Range': ct.FlowKind.Params,
		},
		input_sockets_optional={
			'σ Order': True,
			'σ Range': True,
		},
	)
	def compute_params(self, props, input_sockets) -> td.Box:
		layers = input_sockets['Layers']

		has_layers = not ct.FlowSignal.check(layers)

		active_socket_set = props['active_socket_set']
		if has_layers:
			# Simple PML
			if active_socket_set == 'Simple':
				return layers

			# Full PML
			sig_order = input_sockets['σ Order']
			sig_range = input_sockets['σ Range']

			has_sig_order = not ct.FlowSignal.check(sig_order)
			has_sig_range = not ct.FlowSignal.check(sig_range)

			if has_sig_order and has_sig_range:
				return layers | sig_order | sig_range
		return ct.FlowSignal.FlowPending


####################
# - Blender Registration
####################
BL_REGISTER = [
	AdiabAbsorbBoundCondNode,
]
BL_NODES = {ct.NodeType.AdiabAbsorbBoundCond: (ct.NodeCategory.MAXWELLSIM_BOUNDS)}
