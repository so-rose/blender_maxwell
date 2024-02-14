from . import fdtd_simulation

BL_REGISTER = [
	*fdtd_simulation.BL_REGISTER,
]
BL_NODES = {
	**fdtd_simulation.BL_NODES,
}
