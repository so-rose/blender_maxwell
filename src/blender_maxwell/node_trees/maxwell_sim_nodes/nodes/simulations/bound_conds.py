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

"""Implements `BoundCondsNode`."""

import typing as typ

import tidy3d as td

from blender_maxwell.utils import logger

from ... import contracts as ct
from ... import sockets
from .. import base, events

log = logger.get(__name__)


SSA = ct.SimSpaceAxis
FK = ct.FlowKind
FS = ct.FlowSignal


class BoundCondsNode(base.MaxwellSimNode):
	"""Provides a hub for joining custom simulation domain boundary conditions by-axis."""

	node_type = ct.NodeType.BoundConds
	bl_label = 'Bound Conds'

	####################
	# - Sockets
	####################
	input_socket_sets: typ.ClassVar = {
		'XYZ': {
			'X': sockets.MaxwellBoundCondSocketDef(allow_axes={SSA.X}),
			'Y': sockets.MaxwellBoundCondSocketDef(allow_axes={SSA.Y}),
			'Z': sockets.MaxwellBoundCondSocketDef(allow_axes={SSA.Z}),
		},
		'±X | YZ': {
			'+X': sockets.MaxwellBoundCondSocketDef(allow_axes={SSA.X}),
			'-X': sockets.MaxwellBoundCondSocketDef(allow_axes={SSA.X}),
			'Y': sockets.MaxwellBoundCondSocketDef(allow_axes={SSA.Y}),
			'Z': sockets.MaxwellBoundCondSocketDef(allow_axes={SSA.Z}),
		},
		'X | ±Y | Z': {
			'X': sockets.MaxwellBoundCondSocketDef(allow_axes={SSA.X}),
			'+Y': sockets.MaxwellBoundCondSocketDef(allow_axes={SSA.Y}),
			'-Y': sockets.MaxwellBoundCondSocketDef(allow_axes={SSA.Y}),
			'Z': sockets.MaxwellBoundCondSocketDef(allow_axes={SSA.Z}),
		},
		'XY | ±Z': {
			'X': sockets.MaxwellBoundCondSocketDef(allow_axes={SSA.X}),
			'Y': sockets.MaxwellBoundCondSocketDef(allow_axes={SSA.Y}),
			'+Z': sockets.MaxwellBoundCondSocketDef(allow_axes={SSA.Z}),
			'-Z': sockets.MaxwellBoundCondSocketDef(allow_axes={SSA.Z}),
		},
		'±XY | Z': {
			'+X': sockets.MaxwellBoundCondSocketDef(allow_axes={SSA.X}),
			'-X': sockets.MaxwellBoundCondSocketDef(allow_axes={SSA.X}),
			'+Y': sockets.MaxwellBoundCondSocketDef(allow_axes={SSA.Y}),
			'-Y': sockets.MaxwellBoundCondSocketDef(allow_axes={SSA.Y}),
			'Z': sockets.MaxwellBoundCondSocketDef(allow_axes={SSA.Z}),
		},
		'X | ±YZ': {
			'X': sockets.MaxwellBoundCondSocketDef(allow_axes={SSA.X}),
			'+Y': sockets.MaxwellBoundCondSocketDef(allow_axes={SSA.Y}),
			'-Y': sockets.MaxwellBoundCondSocketDef(allow_axes={SSA.Y}),
			'+Z': sockets.MaxwellBoundCondSocketDef(allow_axes={SSA.Z}),
			'-Z': sockets.MaxwellBoundCondSocketDef(allow_axes={SSA.Z}),
		},
		'±XYZ': {
			'+X': sockets.MaxwellBoundCondSocketDef(allow_axes={SSA.X}),
			'-X': sockets.MaxwellBoundCondSocketDef(allow_axes={SSA.X}),
			'+Y': sockets.MaxwellBoundCondSocketDef(allow_axes={SSA.Y}),
			'-Y': sockets.MaxwellBoundCondSocketDef(allow_axes={SSA.Y}),
			'+Z': sockets.MaxwellBoundCondSocketDef(allow_axes={SSA.Z}),
			'-Z': sockets.MaxwellBoundCondSocketDef(allow_axes={SSA.Z}),
		},
	}
	output_sockets: typ.ClassVar = {
		'BCs': sockets.MaxwellBoundCondsSocketDef(),
	}

	####################
	# - FlowKind.Value
	####################
	@events.computes_output_socket(
		'BCs',
		kind=FK.Value,
		# Loaded
		input_sockets={'X', 'Y', 'Z', '+X', '-X', '+Y', '-Y', '+Z', '-Z'},
		input_sockets_optional={
			'X',
			'Y',
			'Z',
			'+X',
			'-X',
			'+Y',
			'-Y',
			'+Z',
			'-Z',
		},
		outscks_kinds={'BCs': FK.Params},
	)
	def compute_bcs_value(self, input_sockets, output_sockets) -> td.BoundarySpec:
		"""Compute the simulation boundary conditions, by combining the individual input by specified half axis."""
		output_params = output_sockets['BCs']
		has_output_params = not FS.check(output_params)

		# Deduce "Doubledness"
		## -> A "doubled" axis defines the same bound cond both ways
		x = input_sockets['X']
		y = input_sockets['Y']
		z = input_sockets['Z']

		has_doubled_x = not FS.check_single(x, FS.NoFlow)
		has_doubled_y = not FS.check_single(y, FS.NoFlow)
		has_doubled_z = not FS.check_single(z, FS.NoFlow)

		# Deduce +/- of Each Axis
		## -> +/- X
		if has_doubled_x:
			x_pos = input_sockets['X']
			x_neg = input_sockets['X']
		else:
			x_pos = input_sockets['+X']
			x_neg = input_sockets['-X']

		has_x_pos = not FS.check(x_pos)
		has_x_neg = not FS.check(x_neg)

		## -> +/- Y
		if has_doubled_y:
			y_pos = input_sockets['Y']
			y_neg = input_sockets['Y']
		else:
			y_pos = input_sockets['+Y']
			y_neg = input_sockets['-Y']

		has_y_pos = not FS.check(y_pos)
		has_y_neg = not FS.check(y_neg)

		## -> +/- Z
		if has_doubled_z:
			z_pos = input_sockets['Z']
			z_neg = input_sockets['Z']
		else:
			z_pos = input_sockets['+Z']
			z_neg = input_sockets['-Z']

		has_z_pos = not FS.check(z_pos)
		has_z_neg = not FS.check(z_neg)

		if (
			has_x_pos
			and has_x_neg
			and has_y_pos
			and has_y_neg
			and has_z_pos
			and has_z_neg
			and has_output_params
			and not output_params.symbols
		):
			return td.BoundarySpec(
				x=td.Boundary(
					plus=x_pos,
					minus=x_neg,
				),
				y=td.Boundary(
					plus=y_pos,
					minus=y_neg,
				),
				z=td.Boundary(
					plus=z_pos,
					minus=z_neg,
				),
			)
		return FS.FlowPending

	####################
	# - FlowKind.Func
	####################
	@events.computes_output_socket(
		'BCs',
		kind=FK.Func,
		input_sockets={'X', 'Y', 'Z', '+X', '-X', '+Y', '-Y', '+Z', '-Z'},
		input_socket_kinds={
			'X': FK.Func,
			'Y': FK.Func,
			'Z': FK.Func,
			'+X': FK.Func,
			'-X': FK.Func,
			'+Y': FK.Func,
			'-Y': FK.Func,
			'+Z': FK.Func,
			'-Z': FK.Func,
		},
		input_sockets_optional={'X', 'Y', 'Z', '+X', '-X', '+Y', '-Y', '+Z', '-Z'},
	)
	def compute_bcs_func(self, input_sockets) -> ct.ParamsFlow | FS:
		"""Compute the simulation boundary conditions, by combining the individual input by specified half axis."""
		# Deduce "Doubledness"
		## -> A "doubled" axis defines the same bound cond both ways
		x = input_sockets['X']
		y = input_sockets['Y']
		z = input_sockets['Z']

		has_doubled_x = not FS.check_single(x, FS.NoFlow)
		has_doubled_y = not FS.check_single(y, FS.NoFlow)
		has_doubled_z = not FS.check_single(z, FS.NoFlow)

		# Deduce +/- of Each Axis
		## -> +/- X
		if has_doubled_x:
			x_pos = input_sockets['X']
			x_neg = input_sockets['X']
		else:
			x_pos = input_sockets['+X']
			x_neg = input_sockets['-X']

		has_x_pos = not FS.check(x_pos)
		has_x_neg = not FS.check(x_neg)

		## -> +/- Y
		if has_doubled_y:
			y_pos = input_sockets['Y']
			y_neg = input_sockets['Y']
		else:
			y_pos = input_sockets['+Y']
			y_neg = input_sockets['-Y']

		has_y_pos = not FS.check(y_pos)
		has_y_neg = not FS.check(y_neg)

		## -> +/- Z
		if has_doubled_z:
			z_pos = input_sockets['Z']
			z_neg = input_sockets['Z']
		else:
			z_pos = input_sockets['+Z']
			z_neg = input_sockets['-Z']

		has_z_pos = not FS.check(z_pos)
		has_z_neg = not FS.check(z_neg)

		if (
			has_x_pos
			and has_x_neg
			and has_y_pos
			and has_y_neg
			and has_z_pos
			and has_z_neg
		):
			return (x_pos | x_neg | y_pos | y_neg | z_pos | z_neg).compose_within(
				enclosing_func=lambda els: td.BoundarySpec(
					x=td.Boundary(
						plus=els[0],
						minus=els[1],
					),
					y=td.Boundary(
						plus=els[2],
						minus=els[3],
					),
					z=td.Boundary(
						plus=els[4],
						minus=els[5],
					),
				),
				supports_jax=False,
			)
		return FS.FlowPending

	####################
	# - FlowKind.Params
	####################
	@events.computes_output_socket(
		'BCs',
		kind=FK.Params,
		input_sockets={'X', 'Y', 'Z', '+X', '-X', '+Y', '-Y', '+Z', '-Z'},
		input_socket_kinds={
			'X': FK.Params,
			'Y': FK.Params,
			'Z': FK.Params,
			'+X': FK.Params,
			'-X': FK.Params,
			'+Y': FK.Params,
			'-Y': FK.Params,
			'+Z': FK.Params,
			'-Z': FK.Params,
		},
		input_sockets_optional={'X', 'Y', 'Z', '+X', '-X', '+Y', '-Y', '+Z', '-Z'},
	)
	def compute_bcs_params(self, input_sockets) -> ct.ParamsFlow | FS:
		"""Compute the simulation boundary conditions, by combining the individual input by specified half axis."""
		# Deduce "Doubledness"
		## -> A "doubled" axis defines the same bound cond both ways
		x = input_sockets['X']
		y = input_sockets['Y']
		z = input_sockets['Z']

		has_doubled_x = not FS.check_single(x, FS.NoFlow)
		has_doubled_y = not FS.check_single(y, FS.NoFlow)
		has_doubled_z = not FS.check_single(z, FS.NoFlow)

		# Deduce +/- of Each Axis
		## -> +/- X
		if has_doubled_x:
			x_pos = input_sockets['X']
			x_neg = input_sockets['X']
		else:
			x_pos = input_sockets['+X']
			x_neg = input_sockets['-X']

		has_x_pos = not FS.check(x_pos)
		has_x_neg = not FS.check(x_neg)

		## -> +/- Y
		if has_doubled_y:
			y_pos = input_sockets['Y']
			y_neg = input_sockets['Y']
		else:
			y_pos = input_sockets['+Y']
			y_neg = input_sockets['-Y']

		has_y_pos = not FS.check(y_pos)
		has_y_neg = not FS.check(y_neg)

		## -> +/- Z
		if has_doubled_z:
			z_pos = input_sockets['Z']
			z_neg = input_sockets['Z']
		else:
			z_pos = input_sockets['+Z']
			z_neg = input_sockets['-Z']

		has_z_pos = not FS.check(z_pos)
		has_z_neg = not FS.check(z_neg)

		if (
			has_x_pos
			and has_x_neg
			and has_y_pos
			and has_y_neg
			and has_z_pos
			and has_z_neg
		):
			return x_pos | x_neg | y_pos | y_neg | z_pos | z_neg
		return FS.FlowPending


####################
# - Blender Registration
####################
BL_REGISTER = [
	BoundCondsNode,
]
BL_NODES = {ct.NodeType.BoundConds: (ct.NodeCategory.MAXWELLSIM_SIMS)}
