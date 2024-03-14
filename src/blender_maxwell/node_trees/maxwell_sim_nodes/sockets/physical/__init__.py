from . import unit_system
PhysicalUnitSystemSocketDef = unit_system.PhysicalUnitSystemSocketDef

from . import time
PhysicalTimeSocketDef = time.PhysicalTimeSocketDef

from . import angle
PhysicalAngleSocketDef = angle.PhysicalAngleSocketDef

from . import length
from . import area
from . import volume
PhysicalLengthSocketDef = length.PhysicalLengthSocketDef
PhysicalAreaSocketDef = area.PhysicalAreaSocketDef
PhysicalVolumeSocketDef = volume.PhysicalVolumeSocketDef

from . import point_3d
PhysicalPoint3DSocketDef = point_3d.PhysicalPoint3DSocketDef

from . import size_3d
PhysicalSize3DSocketDef = size_3d.PhysicalSize3DSocketDef

from . import mass
PhysicalMassSocketDef = mass.PhysicalMassSocketDef

from . import speed
from . import accel_scalar
from . import force_scalar
PhysicalSpeedSocketDef = speed.PhysicalSpeedSocketDef
PhysicalAccelScalarSocketDef = accel_scalar.PhysicalAccelScalarSocketDef
PhysicalForceScalarSocketDef = force_scalar.PhysicalForceScalarSocketDef

from . import pol
PhysicalPolSocketDef = pol.PhysicalPolSocketDef

from . import freq
PhysicalFreqSocketDef = freq.PhysicalFreqSocketDef


BL_REGISTER = [
	*unit_system.BL_REGISTER,
	
	*time.BL_REGISTER,
	
	*angle.BL_REGISTER,
	
	*length.BL_REGISTER,
	*area.BL_REGISTER,
	*volume.BL_REGISTER,
	
	*point_3d.BL_REGISTER,
	
	*size_3d.BL_REGISTER,
	
	*mass.BL_REGISTER,
	
	*speed.BL_REGISTER,
	*accel_scalar.BL_REGISTER,
	*force_scalar.BL_REGISTER,
	
	*pol.BL_REGISTER,
	
	*freq.BL_REGISTER,
]
