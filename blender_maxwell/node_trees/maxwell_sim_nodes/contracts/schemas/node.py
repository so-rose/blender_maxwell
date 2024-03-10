import typing as typ

import pydantic as pyd

from ..bl import ManagedObjName, SocketName
from ..managed_obj_type import ManagedObjType

class MaxwellSimNode(typ.Protocol):
	
