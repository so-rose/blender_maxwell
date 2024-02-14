from . import point_dipole_source

BL_REGISTER = [
	*point_dipole_source.BL_REGISTER,
]
BL_NODES = {
	**point_dipole_source.BL_NODES,
}
