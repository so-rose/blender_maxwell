from . import complex_2d_vector, real_2d_vector

Real2DVectorSocketDef = real_2d_vector.Real2DVectorSocketDef
Complex2DVectorSocketDef = complex_2d_vector.Complex2DVectorSocketDef

from . import complex_3d_vector, integer_3d_vector, real_3d_vector

Integer3DVectorSocketDef = integer_3d_vector.Integer3DVectorSocketDef
Real3DVectorSocketDef = real_3d_vector.Real3DVectorSocketDef
Complex3DVectorSocketDef = complex_3d_vector.Complex3DVectorSocketDef


BL_REGISTER = [
	*real_2d_vector.BL_REGISTER,
	*complex_2d_vector.BL_REGISTER,
	*integer_3d_vector.BL_REGISTER,
	*real_3d_vector.BL_REGISTER,
	*complex_3d_vector.BL_REGISTER,
]
