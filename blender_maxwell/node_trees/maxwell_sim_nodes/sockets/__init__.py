from . import basic
AnySocketDef = basic.AnySocketDef
TextSocketDef = basic.TextSocketDef
FilePathSocketDef = basic.FilePathSocketDef

from . import number
RealNumberSocketDef = number.RealNumberSocketDef
ComplexNumberSocketDef = number.ComplexNumberSocketDef

from . import physical
PhysicalAreaSocketDef = physical.PhysicalAreaSocketDef

from . import maxwell
MaxwellBoundSocketDef = maxwell.MaxwellBoundSocketDef
MaxwellFDTDSimSocketDef = maxwell.MaxwellFDTDSimSocketDef
MaxwellMediumSocketDef = maxwell.MaxwellMediumSocketDef
MaxwellSourceSocketDef = maxwell.MaxwellSourceSocketDef
MaxwellStructureSocketDef = maxwell.MaxwellStructureSocketDef

BL_REGISTER = [
	*basic.BL_REGISTER,
	*number.BL_REGISTER,
	*physical.BL_REGISTER,
	*maxwell.BL_REGISTER,
]
