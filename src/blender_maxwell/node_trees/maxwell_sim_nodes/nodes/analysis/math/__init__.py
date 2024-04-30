from . import filter_math, map_math, operate_math  # , #reduce_math, transform_math

BL_REGISTER = [
	*operate_math.BL_REGISTER,
	*map_math.BL_REGISTER,
	*filter_math.BL_REGISTER,
	# *reduce_math.BL_REGISTER,
	# *transform_math.BL_REGISTER,
]
BL_NODES = {
	**operate_math.BL_NODES,
	**map_math.BL_NODES,
	**filter_math.BL_NODES,
	# **reduce_math.BL_NODES,
	# **transform_math.BL_NODES,
}
