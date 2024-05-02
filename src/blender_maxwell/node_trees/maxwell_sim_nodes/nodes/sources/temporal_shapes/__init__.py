# from . import expr_temporal_shape, pulse_temporal_shape, wave_temporal_shape
from . import pulse_temporal_shape, wave_temporal_shape

BL_REGISTER = [
	*pulse_temporal_shape.BL_REGISTER,
	*wave_temporal_shape.BL_REGISTER,
	# *expr_temporal_shape.BL_REGISTER,
]
BL_NODES = {
	**pulse_temporal_shape.BL_NODES,
	**wave_temporal_shape.BL_NODES,
	# **expr_temporal_shape.BL_NODES,
}
