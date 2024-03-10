#from . import object_structure
#from . import geonodes_structure
#from . import scripted_structure

from . import primitives

BL_REGISTER = [
#	*object_structure.BL_REGISTER,
#	*geonodes_structure.BL_REGISTER,
#	*scripted_structure.BL_REGISTER,
	*primitives.BL_REGISTER,
]
BL_NODES = {
#	**object_structure.BL_NODES,
#	**geonodes_structure.BL_NODES,
#	**scripted_structure.BL_NODES,
	**primitives.BL_NODES,
}
