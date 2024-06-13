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
	eme_solver,
	fdtd_solver,
	mode_solver,
)

BL_REGISTER = [
	*fdtd_solver.BL_REGISTER,
	*mode_solver.BL_REGISTER,
	*eme_solver.BL_REGISTER,
]
BL_NODES = {
	**fdtd_solver.BL_NODES,
	**mode_solver.BL_NODES,
	**eme_solver.BL_NODES,
}
