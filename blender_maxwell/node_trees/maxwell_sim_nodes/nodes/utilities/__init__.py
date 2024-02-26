from . import combine
#from . import separate

from . import math
from . import operations
from . import converter

BL_REGISTER = [
	*combine.BL_REGISTER,
	#*separate.BL_REGISTER,
	
	*converter.BL_REGISTER,
	*math.BL_REGISTER,
	*operations.BL_REGISTER,
]
BL_NODES = {
	**combine.BL_NODES,
	#**separate.BL_NODES,
	
	**converter.BL_NODES,
	**math.BL_NODES,
	**operations.BL_NODES,
}
