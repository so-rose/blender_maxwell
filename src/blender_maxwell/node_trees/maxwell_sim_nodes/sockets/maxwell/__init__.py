from . import bound_cond
from . import bound_conds
MaxwellBoundCondSocketDef = bound_cond.MaxwellBoundCondSocketDef
MaxwellBoundCondsSocketDef = bound_conds.MaxwellBoundCondsSocketDef

from . import medium
from . import medium_non_linearity
MaxwellMediumSocketDef = medium.MaxwellMediumSocketDef
MaxwellMediumNonLinearitySocketDef = medium_non_linearity.MaxwellMediumNonLinearitySocketDef

from . import source
from . import temporal_shape
MaxwellSourceSocketDef = source.MaxwellSourceSocketDef
MaxwellTemporalShapeSocketDef = temporal_shape.MaxwellTemporalShapeSocketDef

from . import structure
MaxwellStructureSocketDef = structure.MaxwellStructureSocketDef

from . import monitor
MaxwellMonitorSocketDef = monitor.MaxwellMonitorSocketDef

from . import fdtd_sim
from . import fdtd_sim_data
from . import sim_grid
from . import sim_grid_axis
from . import sim_domain
MaxwellFDTDSimSocketDef = fdtd_sim.MaxwellFDTDSimSocketDef
MaxwellFDTDSimDataSocketDef = fdtd_sim_data.MaxwellFDTDSimDataSocketDef
MaxwellSimGridSocketDef = sim_grid.MaxwellSimGridSocketDef
MaxwellSimGridAxisSocketDef = sim_grid_axis.MaxwellSimGridAxisSocketDef
MaxwellSimDomainSocketDef = sim_domain.MaxwellSimDomainSocketDef


BL_REGISTER = [
	*bound_cond.BL_REGISTER,
	*bound_conds.BL_REGISTER,
	*medium.BL_REGISTER,
	*medium_non_linearity.BL_REGISTER,
	*source.BL_REGISTER,
	*temporal_shape.BL_REGISTER,
	*structure.BL_REGISTER,
	*monitor.BL_REGISTER,
	*fdtd_sim.BL_REGISTER,
	*fdtd_sim_data.BL_REGISTER,
	*sim_grid.BL_REGISTER,
	*sim_grid_axis.BL_REGISTER,
	*sim_domain.BL_REGISTER,
]
