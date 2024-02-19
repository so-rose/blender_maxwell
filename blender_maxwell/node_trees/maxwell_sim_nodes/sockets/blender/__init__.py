from . import object_socket
from . import collection_socket
BlenderObjectSocketDef = object_socket.BlenderObjectSocketDef
BlenderCollectionSocketDef = collection_socket.BlenderCollectionSocketDef

from . import image_socket
from . import volume_socket
BlenderImageSocketDef = image_socket.BlenderImageSocketDef
BlenderVolumeSocketDef = volume_socket.BlenderVolumeSocketDef

from . import geonodes_socket
from . import text_socket
BlenderGeoNodesSocketDef = geonodes_socket.BlenderGeoNodesSocketDef
BlenderTextSocketDef = text_socket.BlenderTextSocketDef


BL_REGISTER = [
	*object_socket.BL_REGISTER,
	*collection_socket.BL_REGISTER,
	
	*image_socket.BL_REGISTER,
	*volume_socket.BL_REGISTER,
	
	*geonodes_socket.BL_REGISTER,
	*text_socket.BL_REGISTER,
]
