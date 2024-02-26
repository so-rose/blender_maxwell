from . import basic
AnySocketDef = basic.AnySocketDef
BoolSocketDef = basic.BoolSocketDef
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
PhysicalVacWLSocketDef = physical.PhysicalVacWLSocketDef
PhysicalSpecRelPermDistSocketDef = physical.PhysicalSpecRelPermDistSocketDef
PhysicalSpecPowerDistSocketDef = physical.PhysicalSpecPowerDistSocketDef

from . import blender
BlenderObjectSocketDef = blender.BlenderObjectSocketDef
BlenderCollectionSocketDef = blender.BlenderCollectionSocketDef
BlenderImageSocketDef = blender.BlenderImageSocketDef
BlenderVolumeSocketDef = blender.BlenderVolumeSocketDef
BlenderGeoNodesSocketDef = blender.BlenderGeoNodesSocketDef
BlenderTextSocketDef = blender.BlenderTextSocketDef
BlenderPreviewTargetSocketDef = blender.BlenderPreviewTargetSocketDef

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
