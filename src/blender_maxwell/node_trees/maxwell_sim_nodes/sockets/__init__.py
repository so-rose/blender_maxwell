from blender_maxwell.utils import logger

from .. import contracts as ct
from . import basic, blender, expr, maxwell, physical, tidy3d
from .scan_socket_defs import scan_for_socket_defs

log = logger.get(__name__)
sockets_modules = [basic, blender, expr, maxwell, physical, tidy3d]

####################
# - Scan for SocketDefs
####################
SOCKET_DEFS = {}
for sockets_module in sockets_modules:
	SOCKET_DEFS |= scan_for_socket_defs(sockets_module)

# Set Global Names from SOCKET_DEFS
## SOCKET_DEFS values are the classes themselves, which always have a __name__.
for socket_def_type in SOCKET_DEFS.values():
	globals()[socket_def_type.__name__] = socket_def_type

# Validate SocketType -> SocketDef
## All SocketTypes should have a SocketDef
for socket_type in ct.SocketType:
	if (
		globals().get(socket_type.value.removesuffix('SocketType') + 'SocketDef')
		is None
	):
		log.warning('Missing SocketDef for %s', socket_type.value)


####################
# - Exports
####################
BL_REGISTER = [
	*basic.BL_REGISTER,
	*blender.BL_REGISTER,
	*expr.BL_REGISTER,
	*maxwell.BL_REGISTER,
	*physical.BL_REGISTER,
	*tidy3d.BL_REGISTER,
]

__all__ = [
	'basic',
	'blender',
	'expr',
	'maxwell',
	'physical',
	'tidy3d',
] + [socket_def_type.__name__ for socket_def_type in SOCKET_DEFS.values()]
