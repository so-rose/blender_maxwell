# from . import sim_grid
# from . import sim_grid_axes
from . import combine, fdtd_sim, sim_domain

BL_REGISTER = [
	*combine.BL_REGISTER,
	*sim_domain.BL_REGISTER,
	# *sim_grid.BL_REGISTER,
	# *sim_grid_axes.BL_REGISTER,
	*fdtd_sim.BL_REGISTER,
]
BL_NODES = {
	**combine.BL_NODES,
	**sim_domain.BL_NODES,
	# **sim_grid.BL_NODES,
	# **sim_grid_axes.BL_NODES,
	**fdtd_sim.BL_NODES,
}
