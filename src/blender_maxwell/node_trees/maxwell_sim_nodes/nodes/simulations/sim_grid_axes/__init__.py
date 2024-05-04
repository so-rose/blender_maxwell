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

from . import (
	array_sim_grid_axis,
	automatic_sim_grid_axis,
	manual_sim_grid_axis,
	uniform_sim_grid_axis,
)

BL_REGISTER = [
	*automatic_sim_grid_axis.BL_REGISTER,
	*manual_sim_grid_axis.BL_REGISTER,
	*uniform_sim_grid_axis.BL_REGISTER,
	*array_sim_grid_axis.BL_REGISTER,
]
BL_NODES = {
	**automatic_sim_grid_axis.BL_NODES,
	**manual_sim_grid_axis.BL_NODES,
	**uniform_sim_grid_axis.BL_NODES,
	**array_sim_grid_axis.BL_NODES,
}
