from . import object as object_socket
from . import collection
BlenderObjectSocketDef = object_socket.BlenderObjectSocketDef
BlenderCollectionSocketDef = collection.BlenderCollectionSocketDef

from . import image
BlenderImageSocketDef = image.BlenderImageSocketDef

from . import geonodes
from . import text
BlenderGeoNodesSocketDef = geonodes.BlenderGeoNodesSocketDef
BlenderTextSocketDef = text.BlenderTextSocketDef

BL_REGISTER = [
	*object_socket.BL_REGISTER,
	*collection.BL_REGISTER,
	
	*text.BL_REGISTER,
	*image.BL_REGISTER,
	*geonodes.BL_REGISTER,
]
