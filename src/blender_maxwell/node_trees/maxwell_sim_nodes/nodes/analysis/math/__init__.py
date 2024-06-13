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

from . import filter_math, map_math, operate_math, reduce_math, transform_math

BL_REGISTER = [
	*operate_math.BL_REGISTER,
	*map_math.BL_REGISTER,
	*filter_math.BL_REGISTER,
	*reduce_math.BL_REGISTER,
	*transform_math.BL_REGISTER,
]
BL_NODES = {
	**operate_math.BL_NODES,
	**map_math.BL_NODES,
	**filter_math.BL_NODES,
	**reduce_math.BL_NODES,
	**transform_math.BL_NODES,
}
