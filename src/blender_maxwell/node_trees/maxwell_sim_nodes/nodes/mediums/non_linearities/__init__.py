from . import (
    add_non_linearity,
    chi_3_susceptibility_non_linearity,
    kerr_non_linearity,
    two_photon_absorption_non_linearity,
)

BL_REGISTER = [
	*add_non_linearity.BL_REGISTER,
	*chi_3_susceptibility_non_linearity.BL_REGISTER,
	*kerr_non_linearity.BL_REGISTER,
	*two_photon_absorption_non_linearity.BL_REGISTER,
]
BL_NODES = {
	**add_non_linearity.BL_NODES,
	**chi_3_susceptibility_non_linearity.BL_NODES,
	**kerr_non_linearity.BL_NODES,
	**two_photon_absorption_non_linearity.BL_NODES,
}
