from . import (
	# astigmatic_gaussian_beam_source,
	# gaussian_beam_source,
	# plane_wave_source,
	# point_dipole_source,
	temporal_shapes,
	# tfsf_source,
	# uniform_current_source,
)

BL_REGISTER = [
	*temporal_shapes.BL_REGISTER,
	#*point_dipole_source.BL_REGISTER,
	# *uniform_current_source.BL_REGISTER,
	#*plane_wave_source.BL_REGISTER,
	# *gaussian_beam_source.BL_REGISTER,
	# *astigmatic_gaussian_beam_source.BL_REGISTER,
	# *tfsf_source.BL_REGISTER,
]
BL_NODES = {
	**temporal_shapes.BL_NODES,
	#**point_dipole_source.BL_NODES,
	# **uniform_current_source.BL_NODES,
	#**plane_wave_source.BL_NODES,
	# **gaussian_beam_source.BL_NODES,
	# **astigmatic_gaussian_beam_source.BL_NODES,
	# **tfsf_source.BL_NODES,
}
