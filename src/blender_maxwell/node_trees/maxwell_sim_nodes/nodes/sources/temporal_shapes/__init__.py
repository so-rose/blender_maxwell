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

# from . import expr_temporal_shape, pulse_temporal_shape, wave_temporal_shape
from . import pulse_temporal_shape, wave_temporal_shape

BL_REGISTER = [
	*pulse_temporal_shape.BL_REGISTER,
	*wave_temporal_shape.BL_REGISTER,
	# *expr_temporal_shape.BL_REGISTER,
]
BL_NODES = {
	**pulse_temporal_shape.BL_NODES,
	**wave_temporal_shape.BL_NODES,
	# **expr_temporal_shape.BL_NODES,
}
