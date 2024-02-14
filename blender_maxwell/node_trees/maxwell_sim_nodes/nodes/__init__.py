from . import inputs
from . import outputs
from . import sources
from . import mediums
from . import simulations
from . import structures

BL_REGISTER = [
	*inputs.BL_REGISTER,
	*outputs.BL_REGISTER,
	*mediums.BL_REGISTER,
	*sources.BL_REGISTER,
	*simulations.BL_REGISTER,
	*structures.BL_REGISTER,
]
BL_NODES = {
	**inputs.BL_NODES,
	**outputs.BL_NODES,
	**sources.BL_NODES,
	**mediums.BL_NODES,
	**simulations.BL_NODES,
	**structures.BL_NODES,
}
