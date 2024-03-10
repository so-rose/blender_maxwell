import sympy as sp
sp.printing.str.StrPrinter._default_settings['abbrev'] = True
## In this tree, all Sympy unit printing must be abbreviated.
## By configuring this in __init__.py, we guarantee it for all subimports.
## (Unless, elsewhere, this setting is changed. Be careful!)

from . import sockets
from . import node_tree
from . import nodes
from . import categories

BL_REGISTER = [
	*sockets.BL_REGISTER,
	*node_tree.BL_REGISTER,
	*nodes.BL_REGISTER,
	*categories.BL_REGISTER,
]

BL_NODE_CATEGORIES = [
	*categories.BL_NODE_CATEGORIES,
]
