from . import install_deps, uninstall_deps, manage_pydeps

BL_REGISTER = [
	*install_deps.BL_REGISTER,
	*uninstall_deps.BL_REGISTER,
	*manage_pydeps.BL_REGISTER,
]

BL_HOTKEYS = [
	*install_deps.BL_HOTKEYS,
	*uninstall_deps.BL_HOTKEYS,
	*manage_pydeps.BL_HOTKEYS,
]
