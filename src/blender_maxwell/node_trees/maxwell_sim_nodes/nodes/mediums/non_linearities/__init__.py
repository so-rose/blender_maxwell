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
	add_non_linearity,
	chi_3_susceptibility_non_linearity,
	kerr_non_linearity,
)

BL_REGISTER = [
	*add_non_linearity.BL_REGISTER,
	*chi_3_susceptibility_non_linearity.BL_REGISTER,
	*kerr_non_linearity.BL_REGISTER,
	# *two_photon_absorption_non_linearity.BL_REGISTER,
]
BL_NODES = {
	**add_non_linearity.BL_NODES,
	**chi_3_susceptibility_non_linearity.BL_NODES,
	**kerr_non_linearity.BL_NODES,
	# **two_photon_absorption_non_linearity.BL_NODES,
}
