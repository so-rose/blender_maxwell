from . import map_math, filter_math, reduce_math, operate_math

BL_REGISTER = [
	*map_math.BL_REGISTER,
	*filter_math.BL_REGISTER,
	*reduce_math.BL_REGISTER,
	*operate_math.BL_REGISTER,
]
BL_NODES = {
	**map_math.BL_NODES,
	**filter_math.BL_NODES,
	**reduce_math.BL_NODES,
	**operate_math.BL_NODES,
}
