"""Implements `BoundCondsNode`."""

import typing as typ

import tidy3d as td

from blender_maxwell.utils import logger

from ... import contracts as ct
from ... import sockets
from .. import base, events

log = logger.get(__name__)


class BoundCondsNode(base.MaxwellSimNode):
	"""Provides a hub for joining custom simulation domain boundary conditions by-axis."""

	node_type = ct.NodeType.BoundConds
	bl_label = 'Bound Conds'

	####################
	# - Sockets
	####################
	input_socket_sets: typ.ClassVar = {
		'XYZ': {
			'X': sockets.MaxwellBoundCondSocketDef(),
			'Y': sockets.MaxwellBoundCondSocketDef(),
			'Z': sockets.MaxwellBoundCondSocketDef(),
		},
		'±X | YZ': {
			'+X': sockets.MaxwellBoundCondSocketDef(),
			'-X': sockets.MaxwellBoundCondSocketDef(),
			'Y': sockets.MaxwellBoundCondSocketDef(),
			'Z': sockets.MaxwellBoundCondSocketDef(),
		},
		'X | ±Y | Z': {
			'X': sockets.MaxwellBoundCondSocketDef(),
			'+Y': sockets.MaxwellBoundCondSocketDef(),
			'-Y': sockets.MaxwellBoundCondSocketDef(),
			'Z': sockets.MaxwellBoundCondSocketDef(),
		},
		'XY | ±Z': {
			'X': sockets.MaxwellBoundCondSocketDef(),
			'Y': sockets.MaxwellBoundCondSocketDef(),
			'+Z': sockets.MaxwellBoundCondSocketDef(),
			'-Z': sockets.MaxwellBoundCondSocketDef(),
		},
		'±XY | Z': {
			'+X': sockets.MaxwellBoundCondSocketDef(),
			'-X': sockets.MaxwellBoundCondSocketDef(),
			'+Y': sockets.MaxwellBoundCondSocketDef(),
			'-Y': sockets.MaxwellBoundCondSocketDef(),
			'Z': sockets.MaxwellBoundCondSocketDef(),
		},
		'X | ±YZ': {
			'X': sockets.MaxwellBoundCondSocketDef(),
			'+Y': sockets.MaxwellBoundCondSocketDef(),
			'-Y': sockets.MaxwellBoundCondSocketDef(),
			'+Z': sockets.MaxwellBoundCondSocketDef(),
			'-Z': sockets.MaxwellBoundCondSocketDef(),
		},
		'±XYZ': {
			'+X': sockets.MaxwellBoundCondSocketDef(),
			'-X': sockets.MaxwellBoundCondSocketDef(),
			'+Y': sockets.MaxwellBoundCondSocketDef(),
			'-Y': sockets.MaxwellBoundCondSocketDef(),
			'+Z': sockets.MaxwellBoundCondSocketDef(),
			'-Z': sockets.MaxwellBoundCondSocketDef(),
		},
	}
	output_sockets: typ.ClassVar = {
		'BCs': sockets.MaxwellBoundCondsSocketDef(),
	}

	####################
	# - Output Socket Computation
	####################
	@events.computes_output_socket(
		'BCs',
		input_sockets={'X', 'Y', 'Z', '+X', '-X', '+Y', '-Y', '+Z', '-Z'},
		input_sockets_optional={
			'X': True,
			'Y': True,
			'Z': True,
			'+X': True,
			'-X': True,
			'+Y': True,
			'-Y': True,
			'+Z': True,
			'-Z': True,
		},
	)
	def compute_boundary_conds(self, input_sockets) -> td.BoundarySpec:
		"""Compute the simulation boundary conditions, by combining the individual input by specified half axis."""
		log.debug(
			'%s: Computing Boundary Conditions (Input Sockets = %s)',
			self.sim_node_name,
			str(input_sockets),
		)

		# Deduce "Doubledness"
		## -> A "doubled" axis defines the same bound cond both ways
		has_doubled_x = not ct.FlowSignal.check(input_sockets['X'])
		has_doubled_y = not ct.FlowSignal.check(input_sockets['Y'])
		has_doubled_z = not ct.FlowSignal.check(input_sockets['Z'])

		# Deduce +/- of Each Axis
		## +/- X
		if has_doubled_x:
			x_pos = input_sockets['X']
			x_neg = input_sockets['X']
		else:
			x_pos = input_sockets['+X']
			x_neg = input_sockets['-X']

		## +/- Y
		if has_doubled_y:
			y_pos = input_sockets['Y']
			y_neg = input_sockets['Y']
		else:
			y_pos = input_sockets['+Y']
			y_neg = input_sockets['-Y']

		## +/- Z
		if has_doubled_z:
			z_pos = input_sockets['Z']
			z_neg = input_sockets['Z']
		else:
			z_pos = input_sockets['+Z']
			z_neg = input_sockets['-Z']

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


####################
# - Blender Registration
####################
BL_REGISTER = [
	BoundCondsNode,
]
BL_NODES = {ct.NodeType.BoundConds: (ct.NodeCategory.MAXWELLSIM_BOUNDS)}
