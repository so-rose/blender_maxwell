from . import bound_box_socket
from . import bound_face_socket
MaxwellBoundBoxSocketDef = bound_box_socket.MaxwellBoundBoxSocketDef
MaxwellBoundFaceSocketDef = bound_face_socket.MaxwellBoundFaceSocketDef

from . import medium_socket
from . import medium_non_linearity_socket
MaxwellMediumSocketDef = medium_socket.MaxwellMediumSocketDef
MaxwellMediumNonLinearitySocketDef = medium_non_linearity_socket.MaxwellMediumNonLinearitySocketDef

from . import source_socket
from . import temporal_shape_socket
MaxwellSourceSocketDef = source_socket.MaxwellSourceSocketDef
MaxwellTemporalShapeSocketDef = temporal_shape_socket.MaxwellTemporalShapeSocketDef

from . import structure_socket
MaxwellStructureSocketDef = structure_socket.MaxwellStructureSocketDef

from . import monitor_socket
MaxwellMonitorSocketDef = monitor_socket.MaxwellMonitorSocketDef

from . import fdtd_sim_socket
from . import sim_grid_socket
from . import sim_grid_axis_socket
from . import sim_domain_socket
MaxwellFDTDSimSocketDef = fdtd_sim_socket.MaxwellFDTDSimSocketDef
MaxwellSimGridSocketDef = sim_grid_socket.MaxwellSimGridSocketDef
MaxwellSimGridAxisSocketDef = sim_grid_axis_socket.MaxwellSimGridAxisSocketDef
MaxwellSimDomainSocketDef = sim_domain_socket.MaxwellSimDomainSocketDef


BL_REGISTER = [
	*bound_box_socket.BL_REGISTER,
	*bound_face_socket.BL_REGISTER,
	*medium_socket.BL_REGISTER,
	*medium_non_linearity_socket.BL_REGISTER,
	*source_socket.BL_REGISTER,
	*temporal_shape_socket.BL_REGISTER,
	*structure_socket.BL_REGISTER,
	*monitor_socket.BL_REGISTER,
	*fdtd_sim_socket.BL_REGISTER,
	*sim_grid_socket.BL_REGISTER,
	*sim_grid_axis_socket.BL_REGISTER,
	*sim_domain_socket.BL_REGISTER,
]
