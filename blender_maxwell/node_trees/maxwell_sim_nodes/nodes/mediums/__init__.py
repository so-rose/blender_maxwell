from . import library_medium

from . import pec_medium
from . import isotropic_medium
from . import anisotropic_medium

from . import triple_sellmeier_medium
from . import sellmeier_medium
from . import pole_residue_medium
from . import drude_medium
from . import drude_lorentz_medium
from . import debye_medium

from . import non_linearities

BL_REGISTER = [
	*library_medium.BL_REGISTER,
	
	*pec_medium.BL_REGISTER,
	*isotropic_medium.BL_REGISTER,
	*anisotropic_medium.BL_REGISTER,
	
	*triple_sellmeier_medium.BL_REGISTER,
	*sellmeier_medium.BL_REGISTER,
	*pole_residue_medium.BL_REGISTER,
	*drude_medium.BL_REGISTER,
	*drude_lorentz_medium.BL_REGISTER,
	*debye_medium.BL_REGISTER,
	
	*non_linearities.BL_REGISTER,
]
BL_NODES = {
	**library_medium.BL_NODES,
	
	**pec_medium.BL_NODES,
	**isotropic_medium.BL_NODES,
	**anisotropic_medium.BL_NODES,
	
	**triple_sellmeier_medium.BL_NODES,
	**sellmeier_medium.BL_NODES,
	**pole_residue_medium.BL_NODES,
	**drude_medium.BL_NODES,
	**drude_lorentz_medium.BL_NODES,
	**debye_medium.BL_NODES,
	
	**non_linearities.BL_NODES,
}
