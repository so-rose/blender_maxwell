from . import constants
from . import lists
from . import parameters
from . import scene

BL_REGISTER = [
	*scene.BL_REGISTER,
	*constants.BL_REGISTER,
	*parameters.BL_REGISTER,
	*lists.BL_REGISTER,
]
BL_NODES = {
	**scene.BL_NODES,
	**constants.BL_NODES,
	**parameters.BL_NODES,
	**lists.BL_NODES,
}
