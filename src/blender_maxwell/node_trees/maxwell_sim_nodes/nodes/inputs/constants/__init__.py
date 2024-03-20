# from . import scientific_constant
from . import number_constant

# from . import physical_constant
from . import blender_constant

BL_REGISTER = [
	# *scientific_constant.BL_REGISTER,
	*number_constant.BL_REGISTER,
	# *physical_constant.BL_REGISTER,
	*blender_constant.BL_REGISTER,
]
BL_NODES = {
	# **scientific_constant.BL_NODES,
	**number_constant.BL_NODES,
	# **physical_constant.BL_NODES,
	**blender_constant.BL_NODES,
}
