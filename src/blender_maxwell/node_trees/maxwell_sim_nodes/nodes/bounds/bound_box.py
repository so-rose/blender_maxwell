import tidy3d as td
import sympy as sp
import sympy.physics.units as spu

from ... import contracts as ct
from ... import sockets
from .. import base


class BoundCondsNode(base.MaxwellSimNode):
	node_type = ct.NodeType.BoundConds
	bl_label = 'Bound Box'
	# bl_icon = ...

	####################
	# - Sockets
	####################
	input_sockets = {
		'+X': sockets.MaxwellBoundCondSocketDef(),
		'-X': sockets.MaxwellBoundCondSocketDef(),
		'+Y': sockets.MaxwellBoundCondSocketDef(),
		'-Y': sockets.MaxwellBoundCondSocketDef(),
		'+Z': sockets.MaxwellBoundCondSocketDef(),
		'-Z': sockets.MaxwellBoundCondSocketDef(),
	}
	output_sockets = {
		'BCs': sockets.MaxwellBoundCondsSocketDef(),
	}

	####################
	# - Output Socket Computation
	####################
	@base.computes_output_socket(
		'BCs', input_sockets={'+X', '-X', '+Y', '-Y', '+Z', '-Z'}
	)
	def compute_simulation(self, input_sockets) -> td.BoundarySpec:
		x_pos = input_sockets['+X']
		x_neg = input_sockets['-X']
		y_pos = input_sockets['+Y']
		y_neg = input_sockets['-Y']
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
