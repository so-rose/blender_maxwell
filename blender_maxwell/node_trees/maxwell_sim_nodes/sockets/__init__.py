from . import basic
AnySocketDef = basic.AnySocketDef
TextSocketDef = basic.TextSocketDef
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
PhysicalTimeSocketDef = physical.PhysicalTimeSocketDef
PhysicalAngleSocketDef = physical.PhysicalAngleSocketDef
PhysicalLengthSocketDef = physical.PhysicalLengthSocketDef
PhysicalAreaSocketDef = physical.PhysicalAreaSocketDef
PhysicalVolumeSocketDef = physical.PhysicalVolumeSocketDef
PhysicalMassSocketDef = physical.PhysicalMassSocketDef
PhysicalSpeedSocketDef = physical.PhysicalSpeedSocketDef
PhysicalAccelSocketDef = physical.PhysicalAccelSocketDef
PhysicalForceSocketDef = physical.PhysicalForceSocketDef
PhysicalPolSocketDef = physical.PhysicalPolSocketDef
PhysicalFreqSocketDef = physical.PhysicalFreqSocketDef
PhysicalSpecRelPermDistSocketDef = physical.PhysicalSpecRelPermDistSocketDef
PhysicalSpecPowerDistSocketDef = physical.PhysicalSpecPowerDistSocketDef

from . import blender
BlenderObjectSocketDef = blender.BlenderObjectSocketDef
BlenderCollectionSocketDef = blender.BlenderCollectionSocketDef
BlenderImageSocketDef = blender.BlenderImageSocketDef
BlenderVolumeSocketDef = blender.BlenderVolumeSocketDef
BlenderGeoNodesSocketDef = blender.BlenderGeoNodesSocketDef
BlenderTextSocketDef = blender.BlenderTextSocketDef

from . import maxwell
MaxwellBoundBoxSocketDef = maxwell.MaxwellBoundBoxSocketDef
MaxwellBoundFaceSocketDef = maxwell.MaxwellBoundFaceSocketDef
MaxwellMediumSocketDef = maxwell.MaxwellMediumSocketDef
MaxwellSourceSocketDef = maxwell.MaxwellSourceSocketDef
MaxwellTemporalShapeSocketDef = maxwell.MaxwellTemporalShapeSocketDef
MaxwellStructureSocketDef = maxwell.MaxwellStructureSocketDef
MaxwellMonitorSocketDef = maxwell.MaxwellMonitorSocketDef
MaxwellFDTDSimSocketDef = maxwell.MaxwellFDTDSimSocketDef
MaxwellSimGridSocketDef = maxwell.MaxwellSimGridSocketDef
MaxwellSimGridAxisSocketDef = maxwell.MaxwellSimGridAxisSocketDef

BL_REGISTER = [
	*basic.BL_REGISTER,
	*number.BL_REGISTER,
	*vector.BL_REGISTER,
	*physical.BL_REGISTER,
	*blender.BL_REGISTER,
	*maxwell.BL_REGISTER,
]
