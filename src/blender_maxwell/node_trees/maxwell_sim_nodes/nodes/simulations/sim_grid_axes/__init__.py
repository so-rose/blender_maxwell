from . import automatic_sim_grid_axis
from . import manual_sim_grid_axis
from . import uniform_sim_grid_axis
from . import array_sim_grid_axis

BL_REGISTER = [
	*automatic_sim_grid_axis.BL_REGISTER,
	*manual_sim_grid_axis.BL_REGISTER,
	*uniform_sim_grid_axis.BL_REGISTER,
	*array_sim_grid_axis.BL_REGISTER,
]
BL_NODES = {
	**automatic_sim_grid_axis.BL_NODES,
	**manual_sim_grid_axis.BL_NODES,
	**uniform_sim_grid_axis.BL_NODES,
	**array_sim_grid_axis.BL_NODES,
}
