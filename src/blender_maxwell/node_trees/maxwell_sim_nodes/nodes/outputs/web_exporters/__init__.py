from . import tidy3d_web_exporter, tidy3d_web_runner

BL_REGISTER = [
	*tidy3d_web_exporter.BL_REGISTER,
	*tidy3d_web_runner.BL_REGISTER,
]
BL_NODES = {
	**tidy3d_web_exporter.BL_NODES,
	**tidy3d_web_runner.BL_NODES,
}
