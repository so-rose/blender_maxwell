from . import extract_data, viz

BL_REGISTER = [
	*extract_data.BL_REGISTER,
	*viz.BL_REGISTER,
]
BL_NODES = {
	**extract_data.BL_NODES,
	**viz.BL_NODES,
}
