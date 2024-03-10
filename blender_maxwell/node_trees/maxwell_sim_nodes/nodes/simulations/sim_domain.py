import bpy
import sympy as sp
import sympy.physics.units as spu
import scipy as sc

from ... import contracts as ct
from ... import sockets
from .. import base

class SimDomainNode(base.MaxwellSimNode):
	node_type = ct.NodeType.SimDomain
	bl_label = "Sim Domain"
	
	input_sockets = {
		"Duration": sockets.PhysicalTimeSocketDef(
			default_value = 5 * spu.ps,
			default_unit = spu.ps,
		),
		"Size": sockets.PhysicalSize3DSocketDef(),
		"Grid": sockets.MaxwellSimGridSocketDef(),
		"Ambient Medium": sockets.MaxwellMediumSocketDef(),
	}
	output_sockets = {
		"Domain": sockets.MaxwellSimDomainSocketDef(),
	}
	
	####################
	# - Callbacks
	####################
	@base.computes_output_socket(
		"Domain",
		input_sockets={"Duration", "Size", "Grid", "Ambient Medium"},
	)
	def compute_sim_domain(self, input_sockets: dict) -> sp.Expr:
		if all([
			(_duration := input_sockets["Duration"]),
			(_size := input_sockets["Size"]),
			(grid := input_sockets["Grid"]),
			(medium := input_sockets["Ambient Medium"]),
		]):
			duration = spu.convert_to(_duration, spu.second) / spu.second
			size = tuple(spu.convert_to(_size, spu.um) / spu.um)
			return dict(
				run_time=duration,
				size=size,
				grid_spec=grid,
				medium=medium,
			)

####################
# - Blender Registration
####################
BL_REGISTER = [
	SimDomainNode,
]
BL_NODES = {
	ct.NodeType.SimDomain: (
		ct.NodeCategory.MAXWELLSIM_SIMS
	)
}
