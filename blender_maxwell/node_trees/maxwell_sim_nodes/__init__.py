from . import nodes
from . import categories
from . import socket_types
from . import tree

BL_REGISTER = [
	*tree.BL_REGISTER,
	*socket_types.BL_REGISTER,
	*nodes.BL_REGISTER,
	*categories.BL_REGISTER,
]

BL_NODE_CATEGORIES = [
	*categories.BL_NODE_CATEGORIES,
]
