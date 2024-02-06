import bpy
import tidy3d as td

from .. import types, constants

class FDTDMaxwellSimulationNode(types.MaxwellSimTreeNode, bpy.types.Node):
	bl_idname = types.FDTDMaxwellSimulationNodeType
	bl_label = "FDTD"
	bl_icon = constants.tree_constants.ICON_SIM_SIMULATION
	
	input_sockets = {
		"run_time": ("NodeSocketFloatTime", "Run Time"),
		"ambient_medium": (types.tree_types.MaxwellMediumSocketType, "Ambient Medium"),
		"source": (types.tree_types.MaxwellSourceSocketType, "Source"),
		"structure": (types.tree_types.MaxwellStructureSocketType, "Structure"),
		"bound": (types.tree_types.MaxwellBoundSocketType, "Bound"),
	}
	output_sockets = {
		"fdtd_sim": (types.tree_types.MaxwellFDTDSimSocketType, "FDTD Sim")
	}

	####################
	# - Socket Properties
	####################
	@types.output_socket_cb("fdtd_sim")
	def output_source(self):
		return td.Simulation(
			#center=  ## Default: 0,0,0
			size=(1, 1, 1),  ## PLACEHOLDER
			run_time=self.compute_input("run_time"),
			#structures=[],
			#symmetry=,
			sources=[self.compute_input("source")],
			#boundary_spec=,  ## Default: PML
			monitors=[],
			#grid_spec=,  ## Default: Autogrid
			#shutoff=,  ## Default: 1e-05
			#subpixel=,  ## Default: True
			#normalize_index=,  ## Default: 0
			#courant=,  ## Default: 0.99
		)


####################
# - Blender Registration
####################
BL_REGISTER = [
	FDTDMaxwellSimulationNode,
]
