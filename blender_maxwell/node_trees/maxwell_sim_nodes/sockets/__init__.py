from . import base

from . import basic
AnySocketDef = basic.AnySocketDef
BoolSocketDef = basic.BoolSocketDef
StringSocketDef = basic.StringSocketDef
FilePathSocketDef = basic.FilePathSocketDef

from . import number
IntegerNumberSocketDef = number.IntegerNumberSocketDef
RationalNumberSocketDef = number.RationalNumberSocketDef
RealNumberSocketDef = number.RealNumberSocketDef
ComplexNumberSocketDef = number.ComplexNumberSocketDef

from . import vector
Real2DVectorSocketDef = vector.Real2DVectorSocketDef
Complex2DVectorSocketDef = vector.Complex2DVectorSocketDef
Real3DVectorSocketDef = vector.Real3DVectorSocketDef
Complex3DVectorSocketDef = vector.Complex3DVectorSocketDef

from . import physical
PhysicalUnitSystemSocketDef = physical.PhysicalUnitSystemSocketDef
PhysicalTimeSocketDef = physical.PhysicalTimeSocketDef
PhysicalAngleSocketDef = physical.PhysicalAngleSocketDef
PhysicalLengthSocketDef = physical.PhysicalLengthSocketDef
PhysicalAreaSocketDef = physical.PhysicalAreaSocketDef
PhysicalVolumeSocketDef = physical.PhysicalVolumeSocketDef
PhysicalPoint3DSocketDef = physical.PhysicalPoint3DSocketDef
PhysicalSize3DSocketDef = physical.PhysicalSize3DSocketDef
PhysicalMassSocketDef = physical.PhysicalMassSocketDef
PhysicalSpeedSocketDef = physical.PhysicalSpeedSocketDef
PhysicalAccelScalarSocketDef = physical.PhysicalAccelScalarSocketDef
PhysicalForceScalarSocketDef = physical.PhysicalForceScalarSocketDef
PhysicalPolSocketDef = physical.PhysicalPolSocketDef
PhysicalFreqSocketDef = physical.PhysicalFreqSocketDef

from . import blender
BlenderObjectSocketDef = blender.BlenderObjectSocketDef
BlenderCollectionSocketDef = blender.BlenderCollectionSocketDef
BlenderImageSocketDef = blender.BlenderImageSocketDef
BlenderGeoNodesSocketDef = blender.BlenderGeoNodesSocketDef
BlenderTextSocketDef = blender.BlenderTextSocketDef

from . import maxwell
MaxwellBoundCondSocketDef = maxwell.MaxwellBoundCondSocketDef
MaxwellBoundCondsSocketDef = maxwell.MaxwellBoundCondsSocketDef
MaxwellMediumSocketDef = maxwell.MaxwellMediumSocketDef
MaxwellMediumNonLinearitySocketDef = maxwell.MaxwellMediumNonLinearitySocketDef
MaxwellSourceSocketDef = maxwell.MaxwellSourceSocketDef
MaxwellTemporalShapeSocketDef = maxwell.MaxwellTemporalShapeSocketDef
MaxwellStructureSocketDef = maxwell.MaxwellStructureSocketDef
MaxwellMonitorSocketDef = maxwell.MaxwellMonitorSocketDef
MaxwellFDTDSimSocketDef = maxwell.MaxwellFDTDSimSocketDef
MaxwellSimGridSocketDef = maxwell.MaxwellSimGridSocketDef
MaxwellSimGridAxisSocketDef = maxwell.MaxwellSimGridAxisSocketDef
MaxwellSimDomainSocketDef = maxwell.MaxwellSimDomainSocketDef

from . import tidy3d
Tidy3DCloudTaskSocketDef = tidy3d.Tidy3DCloudTaskSocketDef

BL_REGISTER = [
	*basic.BL_REGISTER,
	*number.BL_REGISTER,
	*vector.BL_REGISTER,
	*physical.BL_REGISTER,
	*blender.BL_REGISTER,
	*maxwell.BL_REGISTER,
	*tidy3d.BL_REGISTER,
]
