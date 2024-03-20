from . import eh_field_monitor
from . import field_power_flux_monitor
# from . import epsilon_tensor_monitor
# from . import diffraction_monitor

BL_REGISTER = [
	*eh_field_monitor.BL_REGISTER,
	*field_power_flux_monitor.BL_REGISTER,
	# *epsilon_tensor_monitor.BL_REGISTER,
	# *diffraction_monitor.BL_REGISTER,
]
BL_NODES = {
	**eh_field_monitor.BL_NODES,
	**field_power_flux_monitor.BL_NODES,
	# **epsilon_tensor_monitor.BL_NODES,
	# **diffraction_monitor.BL_NODES,
}
