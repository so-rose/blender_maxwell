from . import bound_cond_nodes, bound_conds

BL_REGISTER = [
	*bound_conds.BL_REGISTER,
	*bound_cond_nodes.BL_REGISTER,
]
BL_NODES = {
	**bound_conds.BL_NODES,
	**bound_cond_nodes.BL_NODES,
}
