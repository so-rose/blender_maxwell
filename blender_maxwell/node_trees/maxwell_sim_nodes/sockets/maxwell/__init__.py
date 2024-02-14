from . import maxwell_bound_socket
MaxwellBoundSocketDef = maxwell_bound_socket.MaxwellBoundSocketDef

from . import maxwell_fdtd_sim_socket
MaxwellFDTDSimSocketDef = maxwell_fdtd_sim_socket.MaxwellFDTDSimSocketDef

from . import maxwell_medium_socket
MaxwellMediumSocketDef = maxwell_medium_socket.MaxwellMediumSocketDef

from . import maxwell_source_socket
MaxwellSourceSocketDef = maxwell_source_socket.MaxwellSourceSocketDef

from . import maxwell_structure_socket
MaxwellStructureSocketDef = maxwell_structure_socket.MaxwellStructureSocketDef


BL_REGISTER = [
	*maxwell_bound_socket.BL_REGISTER,
	*maxwell_fdtd_sim_socket.BL_REGISTER,
	*maxwell_medium_socket.BL_REGISTER,
	*maxwell_source_socket.BL_REGISTER,
	*maxwell_structure_socket.BL_REGISTER,
]
