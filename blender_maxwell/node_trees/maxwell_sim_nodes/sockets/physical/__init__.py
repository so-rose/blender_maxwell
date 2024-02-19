from . import time_socket
PhysicalTimeSocketDef = time_socket.PhysicalTimeSocketDef

from . import angle_socket
PhysicalAngleSocketDef = angle_socket.PhysicalAngleSocketDef

from . import length_socket
from . import area_socket
from . import volume_socket
PhysicalLengthSocketDef = length_socket.PhysicalLengthSocketDef
PhysicalAreaSocketDef = area_socket.PhysicalAreaSocketDef
PhysicalVolumeSocketDef = volume_socket.PhysicalVolumeSocketDef

from . import mass_socket
PhysicalMassSocketDef = mass_socket.PhysicalMassSocketDef

from . import speed_socket
from . import accel_socket
from . import force_socket
PhysicalSpeedSocketDef = speed_socket.PhysicalSpeedSocketDef
PhysicalAccelSocketDef = accel_socket.PhysicalAccelSocketDef
PhysicalForceSocketDef = force_socket.PhysicalForceSocketDef

from . import pol_socket
PhysicalPolSocketDef = pol_socket.PhysicalPolSocketDef

from . import freq_socket
PhysicalFreqSocketDef = freq_socket.PhysicalFreqSocketDef

from . import spec_rel_permit_dist_socket
from . import spec_power_dist_socket
PhysicalSpecRelPermDistSocketDef = spec_rel_permit_dist_socket.PhysicalSpecRelPermDistSocketDef
PhysicalSpecPowerDistSocketDef = spec_power_dist_socket.PhysicalSpecPowerDistSocketDef


BL_REGISTER = [
	*time_socket.BL_REGISTER,
	
	*angle_socket.BL_REGISTER,
	
	*length_socket.BL_REGISTER,
	*area_socket.BL_REGISTER,
	*volume_socket.BL_REGISTER,
	
	*mass_socket.BL_REGISTER,
	
	*speed_socket.BL_REGISTER,
	*accel_socket.BL_REGISTER,
	*force_socket.BL_REGISTER,
	
	*pol_socket.BL_REGISTER,
	
	*freq_socket.BL_REGISTER,
	*spec_rel_permit_dist_socket.BL_REGISTER,
	*spec_power_dist_socket.BL_REGISTER,
]
