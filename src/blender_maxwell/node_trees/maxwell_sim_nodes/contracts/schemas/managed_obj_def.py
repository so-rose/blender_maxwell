import typing as typ

import pydantic as pyd

from .managed_obj import ManagedObj


class ManagedObjDef(pyd.BaseModel):
	mk: typ.Callable[[str], ManagedObj]
	name_prefix: str = ''
