from . import tree, socket_types, nodes

BL_REGISTER = [
	*tree.BL_REGISTER,
	*socket_types.BL_REGISTER,
	*nodes.BL_REGISTER,
]

BL_NODE_CATEGORIES = [
	*nodes.BL_NODE_CATEGORIES,
]
