# from . import math
from . import combine

# from . import separate

BL_REGISTER = [
	# *math.BL_REGISTER,
	*combine.BL_REGISTER,
	# *separate.BL_REGISTER,
]
BL_NODES = {
	# **math.BL_NODES,
	**combine.BL_NODES,
	# **separate.BL_NODES,
}
