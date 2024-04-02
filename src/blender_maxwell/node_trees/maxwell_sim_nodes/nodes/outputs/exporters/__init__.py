from . import json_file_exporter, tidy3d_web_exporter

BL_REGISTER = [
	*json_file_exporter.BL_REGISTER,
	*tidy3d_web_exporter.BL_REGISTER,
]
BL_NODES = {
	**json_file_exporter.BL_NODES,
	**tidy3d_web_exporter.BL_NODES,
}
