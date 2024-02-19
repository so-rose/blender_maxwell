from . import number_list
from . import physical_list

BL_REGISTER = [
	*number_list.BL_REGISTER,
	*physical_list.BL_REGISTER,
]
BL_NODES = {
	**number_list.BL_NODES,
	**physical_list.BL_NODES,
}
