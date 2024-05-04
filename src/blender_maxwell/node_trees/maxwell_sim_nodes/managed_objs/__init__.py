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

from .base import ManagedObj

# from .managed_bl_empty import ManagedBLEmpty
from .managed_bl_image import ManagedBLImage

# from .managed_bl_collection import ManagedBLCollection
# from .managed_bl_object import ManagedBLObject
from .managed_bl_mesh import ManagedBLMesh

# from .managed_bl_volume import ManagedBLVolume
from .managed_bl_modifier import ManagedBLModifier

__all__ = [
	'ManagedObj',
	#'ManagedBLEmpty',
	'ManagedBLImage',
	#'ManagedBLCollection',
	#'ManagedBLObject',
	'ManagedBLMesh',
	#'ManagedBLVolume',
	'ManagedBLModifier',
]

## REMEMBER: Add the appropriate entry to the bl_cache.DECODER
