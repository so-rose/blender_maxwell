from . import categories, node_tree, nodes, sockets

BL_REGISTER = [
	*sockets.BL_REGISTER,
	*node_tree.BL_REGISTER,
	*nodes.BL_REGISTER,
	*categories.BL_REGISTER,
]
