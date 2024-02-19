from . import number_parameter
from . import physical_parameter

BL_REGISTER = [
	*number_parameter.BL_REGISTER,
	*physical_parameter.BL_REGISTER,
]
BL_NODES = {
	**number_parameter.BL_NODES,
	**physical_parameter.BL_NODES,
}
