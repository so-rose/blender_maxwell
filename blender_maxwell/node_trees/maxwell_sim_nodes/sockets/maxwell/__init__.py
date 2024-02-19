from . import bound_box_socket
from . import bound_face_socket
MaxwellBoundBoxSocketDef = bound_box_socket.MaxwellBoundBoxSocketDef
MaxwellBoundFaceSocketDef = bound_face_socket.MaxwellBoundFaceSocketDef

from . import medium_socket
MaxwellMediumSocketDef = medium_socket.MaxwellMediumSocketDef

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
MaxwellFDTDSimSocketDef = fdtd_sim_socket.MaxwellFDTDSimSocketDef
MaxwellSimGridSocketDef = sim_grid_socket.MaxwellSimGridSocketDef
MaxwellSimGridAxisSocketDef = sim_grid_axis_socket.MaxwellSimGridAxisSocketDef


BL_REGISTER = [
	*bound_box_socket.BL_REGISTER,
	*bound_face_socket.BL_REGISTER,
	*medium_socket.BL_REGISTER,
	*source_socket.BL_REGISTER,
	*temporal_shape_socket.BL_REGISTER,
	*structure_socket.BL_REGISTER,
	*fdtd_sim_socket.BL_REGISTER,
	*sim_grid_socket.BL_REGISTER,
	*sim_grid_axis_socket.BL_REGISTER,
]
