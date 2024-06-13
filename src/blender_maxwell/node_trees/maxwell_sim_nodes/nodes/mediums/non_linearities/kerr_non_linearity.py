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

"""Implements `KerrNonLinearity`."""

import typing as typ

import bpy
import tidy3d as td

from blender_maxwell.utils import logger
from blender_maxwell.utils import sympy_extra as spux

from .... import contracts as ct
from .... import sockets
from ... import base, events

log = logger.get(__name__)

FK = ct.FlowKind
FS = ct.FlowSignal
MT = spux.MathType


class KerrNonLinearity(base.MaxwellSimNode):
	r"""An instantaneous non-linear susceptibility described by a $\chi_3$ parameter.

	The model of field-component does not permit component interactions; therefore, it is only valid when the electric field is predominantly polarized along an axis-aligned direction.
	Additionally, strong non-linearities may suffer from divergence issues, since an iterative local method is used to resolve the relation.
	"""

	node_type = ct.NodeType.KerrNonLinearity
	bl_label = 'Kerr Non-Linearity'

	input_sockets: typ.ClassVar = {
		'n₂': sockets.ExprSocketDef(
			active_kind=FK.Value,
			mathtype=MT.Complex,
		),
	}
	output_sockets: typ.ClassVar = {
		'Non-Linearity': sockets.MaxwellMediumNonLinearitySocketDef(
			active_kind=FK.Func
		),
	}

	####################
	# - UI
	####################
	def draw_info(self, _: bpy.types.Context, layout: bpy.types.UILayout) -> None:
		"""Draw the user interfaces of the node's reported info.

		Parameters:
			layout: UI target for drawing.
		"""
		box = layout.box()
		row = box.row()
		row.alignment = 'CENTER'
		row.label(text='Interpretation')

		# Split
		split = box.split(factor=0.4, align=False)

		## LHS: Parameter Names
		col = split.column()
		col.alignment = 'RIGHT'
		col.label(text='n₂:')

		## RHS: Parameter Units
		col = split.column()
		col.label(text='um² / W')

	####################
	# - FlowKind.Value
	####################
	@events.computes_output_socket(
		'Non-Linearity',
		kind=FK.Value,
		# Loaded
		outscks_kinds={
			'Non-Linearity': {FK.Func, FK.Params},
		},
	)
	def compute_value(self, output_sockets) -> ct.ParamsFlow | FS:
		"""The value realizes the output function w/output parameters."""
		value = events.realize_known(output_sockets['Non-Linearity'])
		if value is not None:
			return value
		return ct.FlowSignal.FlowPending

	####################
	# - FlowKind.Func
	####################
	@events.computes_output_socket(
		'Non-Linearity',
		kind=FK.Func,
		# Loaded
		inscks_kinds={
			'n₂': FK.Func,
		},
	)
	def compute_func(self, input_sockets) -> ct.FuncFlow:
		r"""The function encloses the $\chi_3$ parameter in the nonlinear susceptibility."""
		n2 = input_sockets['n₂']
		return n2.compose_within(lambda _n2: td.KerrNonlinearity(n2=_n2))

	####################
	# - FlowKind.Params
	####################
	@events.computes_output_socket(
		'Non-Linearity',
		kind=FK.Params,
		# Loaded
		inscks_kinds={
			'n₂': FK.Params,
		},
	)
	def compute_params(self, input_sockets) -> ct.FuncFlow:
		r"""The function parameters of the non-linearity are identical to that of the $\chi_3$ parameter."""
		return input_sockets['n₂']


####################
# - Blender Registration
####################
BL_REGISTER = [
	KerrNonLinearity,
]
BL_NODES = {
	ct.NodeType.KerrNonLinearity: (ct.NodeCategory.MAXWELLSIM_MEDIUMS_NONLINEARITIES)
}
