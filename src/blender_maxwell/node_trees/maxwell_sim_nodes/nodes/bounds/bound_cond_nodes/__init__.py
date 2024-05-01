from . import (
    #absorbing_bound_cond,
    #bloch_bound_cond,
    pml_bound_cond,
)

BL_REGISTER = [
	*pml_bound_cond.BL_REGISTER,
	#*bloch_bound_cond.BL_REGISTER,
	#*absorbing_bound_cond.BL_REGISTER,
]
BL_NODES = {
	**pml_bound_cond.BL_NODES,
	#**bloch_bound_cond.BL_NODES,
	#**absorbing_bound_cond.BL_NODES,
}
