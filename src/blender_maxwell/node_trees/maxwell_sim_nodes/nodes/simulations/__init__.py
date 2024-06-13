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
	bound_cond_faces,
	bound_conds,
	fdtd_sim,
	sim_domain,
	sim_grid,
	sim_grid_axes,
)

BL_REGISTER = [
	*fdtd_sim.BL_REGISTER,
	*sim_domain.BL_REGISTER,
	*bound_conds.BL_REGISTER,
	*bound_cond_faces.BL_REGISTER,
	*sim_grid.BL_REGISTER,
	*sim_grid_axes.BL_REGISTER,
]
BL_NODES = {
	**fdtd_sim.BL_NODES,
	**sim_domain.BL_NODES,
	**bound_conds.BL_NODES,
	**bound_cond_faces.BL_NODES,
	**sim_grid.BL_NODES,
	**sim_grid_axes.BL_NODES,
}
