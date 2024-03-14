from . import bound_box
from . import bound_faces

BL_REGISTER = [
	*bound_box.BL_REGISTER,
	*bound_faces.BL_REGISTER,
]
BL_NODES = {
	**bound_box.BL_NODES,
	**bound_faces.BL_NODES,
}
