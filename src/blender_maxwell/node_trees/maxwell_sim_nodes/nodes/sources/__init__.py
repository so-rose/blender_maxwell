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
	# astigmatic_gaussian_beam_source,
	# gaussian_beam_source,
	plane_wave_source,
	point_dipole_source,
	temporal_shapes,
)

BL_REGISTER = [
	*temporal_shapes.BL_REGISTER,
	*point_dipole_source.BL_REGISTER,
	# *uniform_current_source.BL_REGISTER,
	*plane_wave_source.BL_REGISTER,
	# *gaussian_beam_source.BL_REGISTER,
	# *astigmatic_gaussian_beam_source.BL_REGISTER,
	# *tfsf_source.BL_REGISTER,
]
BL_NODES = {
	**temporal_shapes.BL_NODES,
	**point_dipole_source.BL_NODES,
	# **uniform_current_source.BL_NODES,
	**plane_wave_source.BL_NODES,
	# **gaussian_beam_source.BL_NODES,
	# **astigmatic_gaussian_beam_source.BL_NODES,
	# **tfsf_source.BL_NODES,
}
