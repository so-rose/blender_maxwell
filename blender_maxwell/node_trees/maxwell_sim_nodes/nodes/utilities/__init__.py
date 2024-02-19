from . import math
from . import operations

BL_REGISTER = [
	*math.BL_REGISTER,
	*operations.BL_REGISTER,
]
BL_NODES = {
	**math.BL_NODES,
	**operations.BL_NODES,
}
