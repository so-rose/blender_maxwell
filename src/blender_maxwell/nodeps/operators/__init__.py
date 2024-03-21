from . import install_deps, uninstall_deps

BL_REGISTER = [
	*install_deps.BL_REGISTER,
	*uninstall_deps.BL_REGISTER,
]

BL_KEYMAP_ITEM_DEFS = [
	*install_deps.BL_KEYMAP_ITEM_DEFS,
	*uninstall_deps.BL_KEYMAP_ITEM_DEFS,
]


__all__ = []
