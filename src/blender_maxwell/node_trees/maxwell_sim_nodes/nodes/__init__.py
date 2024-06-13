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
	analysis,
	inputs,
	mediums,
	monitors,
	outputs,
	simulations,
	solvers,
	sources,
	structures,
	utilities,
)

BL_REGISTER = [
	*analysis.BL_REGISTER,
	*utilities.BL_REGISTER,
	*inputs.BL_REGISTER,
	*solvers.BL_REGISTER,
	*outputs.BL_REGISTER,
	*sources.BL_REGISTER,
	*mediums.BL_REGISTER,
	*structures.BL_REGISTER,
	*monitors.BL_REGISTER,
	*simulations.BL_REGISTER,
]
BL_NODES = {
	**analysis.BL_NODES,
	**utilities.BL_NODES,
	**inputs.BL_NODES,
	**solvers.BL_NODES,
	**outputs.BL_NODES,
	**sources.BL_NODES,
	**mediums.BL_NODES,
	**structures.BL_NODES,
	**monitors.BL_NODES,
	**simulations.BL_NODES,
}
