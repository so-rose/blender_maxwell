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
	blender_constant,
	expr_constant,
	number_constant,
	physical_constant,
	scientific_constant,
)

BL_REGISTER = [
	*expr_constant.BL_REGISTER,
	*scientific_constant.BL_REGISTER,
	*number_constant.BL_REGISTER,
	*physical_constant.BL_REGISTER,
	*blender_constant.BL_REGISTER,
]
BL_NODES = {
	**expr_constant.BL_NODES,
	**scientific_constant.BL_NODES,
	**number_constant.BL_NODES,
	**physical_constant.BL_NODES,
	**blender_constant.BL_NODES,
}
