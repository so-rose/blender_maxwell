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

"""Package providing various tools to handle cached data on Blender objects, especially nodes and node socket classes."""

from ..keyed_cache import KeyedCache, keyed_cache
from .bl_field import BLField
from .bl_prop import BLProp, BLPropType
from .cached_bl_property import CachedBLProperty, cached_bl_property
from .managed_cache import invalidate_nonpersist_instance_id
from .signal import Signal

__all__ = [
	'BLField',
	'BLProp',
	'BLPropType',
	'CachedBLProperty',
	'cached_bl_property',
	'KeyedCache',
	'keyed_cache',
	'invalidate_nonpersist_instance_id',
	'Signal',
]
