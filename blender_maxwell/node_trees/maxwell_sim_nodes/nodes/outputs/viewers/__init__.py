from . import viewer_3d
from . import value_viewer
from . import console_viewer

BL_REGISTER = [
	*viewer_3d.BL_REGISTER,
	*value_viewer.BL_REGISTER,
	*console_viewer.BL_REGISTER,
]
BL_NODES = {
	**viewer_3d.BL_NODES,
	**value_viewer.BL_NODES,
	**console_viewer.BL_NODES,
}
