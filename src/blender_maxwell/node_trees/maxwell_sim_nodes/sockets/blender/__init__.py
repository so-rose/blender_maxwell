# blender_maxwell
# Copyright (C) 2024 blender_maxwell Project Contributors
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

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
