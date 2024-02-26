from . import unit_system

from . import constants
from . import lists
from . import scene

BL_REGISTER = [
	*unit_system.BL_REGISTER,
	
	*scene.BL_REGISTER,
	*constants.BL_REGISTER,
	*lists.BL_REGISTER,
]
BL_NODES = {
	**unit_system.BL_NODES,
	
	**scene.BL_NODES,
	**constants.BL_NODES,
	**lists.BL_NODES,
}
