from . import value_viewer
from . import console_viewer

BL_REGISTER = [
	*value_viewer.BL_REGISTER,
	*console_viewer.BL_REGISTER,
]
BL_NODES = {
	**value_viewer.BL_NODES,
	**console_viewer.BL_NODES,
}
