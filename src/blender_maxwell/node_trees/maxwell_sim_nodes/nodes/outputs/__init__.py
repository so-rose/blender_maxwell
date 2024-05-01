#from . import file_exporters, viewer, web_exporters
from . import viewer

BL_REGISTER = [
	*viewer.BL_REGISTER,
	#*file_exporters.BL_REGISTER,
	#*web_exporters.BL_REGISTER,
]
BL_NODES = {
	**viewer.BL_NODES,
	#**file_exporters.BL_NODES,
	#**web_exporters.BL_NODES,
}
