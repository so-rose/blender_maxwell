from . import install_deps
from . import uninstall_deps
from . import connect_viewer
from . import refresh_td_auth

BL_REGISTER = [
	*install_deps.BL_REGISTER,
	*uninstall_deps.BL_REGISTER,
	*connect_viewer.BL_REGISTER,
	*refresh_td_auth.BL_REGISTER,
]
BL_KMI_REGISTER = [
	*connect_viewer.BL_KMI_REGISTER,
]
