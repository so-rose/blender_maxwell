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
