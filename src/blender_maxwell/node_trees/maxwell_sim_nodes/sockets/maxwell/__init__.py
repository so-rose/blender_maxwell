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
	bound_cond,
	bound_conds,
	fdtd_sim,
	fdtd_sim_data,
	medium,
	medium_non_linearity,
	monitor,
	monitor_data,
	sim_domain,
	sim_grid,
	sim_grid_axis,
	source,
	structure,
	temporal_shape,
)

MaxwellBoundCondSocketDef = bound_cond.MaxwellBoundCondSocketDef
MaxwellBoundCondsSocketDef = bound_conds.MaxwellBoundCondsSocketDef
MaxwellFDTDSimSocketDef = fdtd_sim.MaxwellFDTDSimSocketDef
MaxwellFDTDSimDataSocketDef = fdtd_sim_data.MaxwellFDTDSimDataSocketDef
MaxwellMediumSocketDef = medium.MaxwellMediumSocketDef
MaxwellMediumNonLinearitySocketDef = (
	medium_non_linearity.MaxwellMediumNonLinearitySocketDef
)
MaxwellMonitorSocketDef = monitor.MaxwellMonitorSocketDef
MaxwellMonitorDataSocketDef = monitor_data.MaxwellMonitorDataSocketDef
MaxwellSimDomainSocketDef = sim_domain.MaxwellSimDomainSocketDef
MaxwellSimGridSocketDef = sim_grid.MaxwellSimGridSocketDef
MaxwellSimGridAxisSocketDef = sim_grid_axis.MaxwellSimGridAxisSocketDef
MaxwellSourceSocketDef = source.MaxwellSourceSocketDef
MaxwellStructureSocketDef = structure.MaxwellStructureSocketDef
MaxwellTemporalShapeSocketDef = temporal_shape.MaxwellTemporalShapeSocketDef


BL_REGISTER = [
	*bound_cond.BL_REGISTER,
	*bound_conds.BL_REGISTER,
	*fdtd_sim.BL_REGISTER,
	*fdtd_sim_data.BL_REGISTER,
	*medium.BL_REGISTER,
	*medium_non_linearity.BL_REGISTER,
	*monitor.BL_REGISTER,
	*monitor_data.BL_REGISTER,
	*sim_domain.BL_REGISTER,
	*sim_grid.BL_REGISTER,
	*sim_grid_axis.BL_REGISTER,
	*source.BL_REGISTER,
	*structure.BL_REGISTER,
	*temporal_shape.BL_REGISTER,
]
