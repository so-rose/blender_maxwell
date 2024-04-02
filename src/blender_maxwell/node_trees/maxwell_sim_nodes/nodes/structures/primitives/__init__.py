# from . import cylinder_structure
from . import box_structure, sphere_structure

BL_REGISTER = [
	*box_structure.BL_REGISTER,
	# *cylinder_structure.BL_REGISTER,
	*sphere_structure.BL_REGISTER,
]
BL_NODES = {
	**box_structure.BL_NODES,
	# **cylinder_structure.BL_NODES,
	**sphere_structure.BL_NODES,
}
