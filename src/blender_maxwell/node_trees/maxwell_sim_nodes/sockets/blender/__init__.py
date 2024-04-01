from . import collection, material
from . import object as object_socket

BlenderMaterialSocketDef = material.BlenderMaterialSocketDef
BlenderObjectSocketDef = object_socket.BlenderObjectSocketDef
BlenderCollectionSocketDef = collection.BlenderCollectionSocketDef

from . import image

BlenderImageSocketDef = image.BlenderImageSocketDef

from . import geonodes, text

BlenderGeoNodesSocketDef = geonodes.BlenderGeoNodesSocketDef
BlenderTextSocketDef = text.BlenderTextSocketDef

BL_REGISTER = [
	*material.BL_REGISTER,
	*object_socket.BL_REGISTER,
	*collection.BL_REGISTER,
	*text.BL_REGISTER,
	*image.BL_REGISTER,
	*geonodes.BL_REGISTER,
]
