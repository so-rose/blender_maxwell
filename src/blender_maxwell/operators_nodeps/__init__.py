from . import install_deps
from . import uninstall_deps

BL_REGISTER = [
	*install_deps.BL_REGISTER,
	*uninstall_deps.BL_REGISTER,
]
