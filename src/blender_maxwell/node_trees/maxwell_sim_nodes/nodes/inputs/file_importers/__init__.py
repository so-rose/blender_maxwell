from . import tidy_3d_file_importer

BL_REGISTER = [
	*tidy_3d_file_importer.BL_REGISTER,
]
BL_NODES = {
	**tidy_3d_file_importer.BL_NODES,
}