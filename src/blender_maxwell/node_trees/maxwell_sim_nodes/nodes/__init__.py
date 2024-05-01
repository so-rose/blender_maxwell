from . import (
	analysis,
	bounds,
	inputs,
	# mediums,
	monitors,
	outputs,
	# simulations,
	# sources,
	# structures,
	# utilities,
)

BL_REGISTER = [
	*analysis.BL_REGISTER,
	*inputs.BL_REGISTER,
	*outputs.BL_REGISTER,
	# *sources.BL_REGISTER,
	# *mediums.BL_REGISTER,
	# *structures.BL_REGISTER,
	*bounds.BL_REGISTER,
	*monitors.BL_REGISTER,
	# *simulations.BL_REGISTER,
	# *utilities.BL_REGISTER,
]
BL_NODES = {
	**analysis.BL_NODES,
	**inputs.BL_NODES,
	**outputs.BL_NODES,
	# **sources.BL_NODES,
	# **mediums.BL_NODES,
	# **structures.BL_NODES,
	**bounds.BL_NODES,
	**monitors.BL_NODES,
	# **simulations.BL_NODES,
	# **utilities.BL_NODES,
}
