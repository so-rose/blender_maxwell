from . import sockets
from . import node_tree
from . import nodes
from . import categories

BL_REGISTER = [
	*sockets.BL_REGISTER,
	*node_tree.BL_REGISTER,
	*nodes.BL_REGISTER,
	*categories.BL_REGISTER,
]

BL_NODE_CATEGORIES = [
	*categories.BL_NODE_CATEGORIES,
]
