from . import gaussian_pulse_temporal_shape
from . import continuous_wave_temporal_shape
from . import array_temporal_shape

BL_REGISTER = [
	*gaussian_pulse_temporal_shape.BL_REGISTER,
	*continuous_wave_temporal_shape.BL_REGISTER,
	*array_temporal_shape.BL_REGISTER,
]
BL_NODES = {
	**gaussian_pulse_temporal_shape.BL_NODES,
	**continuous_wave_temporal_shape.BL_NODES,
	**array_temporal_shape.BL_NODES,
}
