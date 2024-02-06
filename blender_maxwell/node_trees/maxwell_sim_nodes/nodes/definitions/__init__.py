#from . import bounds, mediums, simulations, sources, structures
from . import debug
from . import sources
#from . import mediums
#from . import structures
#from . import bounds
from . import simulations

BL_REGISTER = [
	*debug.BL_REGISTER,
	#*bounds.BL_REGISTER,
	#*mediums.BL_REGISTER,
	*sources.BL_REGISTER,
	#*structures.BL_REGISTER,
	*simulations.BL_REGISTER,
]
