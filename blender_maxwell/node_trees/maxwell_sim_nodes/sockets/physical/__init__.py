from . import unit_system_socket
PhysicalUnitSystemSocketDef = unit_system_socket.PhysicalUnitSystemSocketDef

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

from . import point_3d_socket
PhysicalPoint3DSocketDef = point_3d_socket.PhysicalPoint3DSocketDef

from . import size_3d_socket
PhysicalSize3DSocketDef = size_3d_socket.PhysicalSize3DSocketDef

from . import mass_socket
PhysicalMassSocketDef = mass_socket.PhysicalMassSocketDef

from . import speed_socket
from . import accel_scalar_socket
from . import force_scalar_socket
PhysicalSpeedSocketDef = speed_socket.PhysicalSpeedSocketDef
PhysicalAccelScalarSocketDef = accel_scalar_socket.PhysicalAccelScalarSocketDef
PhysicalForceScalarSocketDef = force_scalar_socket.PhysicalForceScalarSocketDef

from . import pol_socket
PhysicalPolSocketDef = pol_socket.PhysicalPolSocketDef

from . import freq_socket
from . import vac_wl_socket
PhysicalFreqSocketDef = freq_socket.PhysicalFreqSocketDef
PhysicalVacWLSocketDef = vac_wl_socket.PhysicalVacWLSocketDef

from . import spec_rel_permit_dist_socket
from . import spec_power_dist_socket
PhysicalSpecRelPermDistSocketDef = spec_rel_permit_dist_socket.PhysicalSpecRelPermDistSocketDef
PhysicalSpecPowerDistSocketDef = spec_power_dist_socket.PhysicalSpecPowerDistSocketDef


BL_REGISTER = [
	*unit_system_socket.BL_REGISTER,
	
	*time_socket.BL_REGISTER,
	
	*angle_socket.BL_REGISTER,
	
	*length_socket.BL_REGISTER,
	*area_socket.BL_REGISTER,
	*volume_socket.BL_REGISTER,
	
	*point_3d_socket.BL_REGISTER,
	
	*size_3d_socket.BL_REGISTER,
	
	*mass_socket.BL_REGISTER,
	
	*speed_socket.BL_REGISTER,
	*accel_scalar_socket.BL_REGISTER,
	*force_scalar_socket.BL_REGISTER,
	
	*pol_socket.BL_REGISTER,
	
	*freq_socket.BL_REGISTER,
	*vac_wl_socket.BL_REGISTER,
	*spec_rel_permit_dist_socket.BL_REGISTER,
	*spec_power_dist_socket.BL_REGISTER,
]
