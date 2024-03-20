from . import pml_bound_face
from . import pec_bound_face
from . import pmc_bound_face

from . import bloch_bound_face
from . import periodic_bound_face
from . import absorbing_bound_face

BL_REGISTER = [
	*pml_bound_face.BL_REGISTER,
	*pec_bound_face.BL_REGISTER,
	*pmc_bound_face.BL_REGISTER,
	*bloch_bound_face.BL_REGISTER,
	*periodic_bound_face.BL_REGISTER,
	*absorbing_bound_face.BL_REGISTER,
]
BL_NODES = {
	**pml_bound_face.BL_NODES,
	**pec_bound_face.BL_NODES,
	**pmc_bound_face.BL_NODES,
	**bloch_bound_face.BL_NODES,
	**periodic_bound_face.BL_NODES,
	**absorbing_bound_face.BL_NODES,
}
