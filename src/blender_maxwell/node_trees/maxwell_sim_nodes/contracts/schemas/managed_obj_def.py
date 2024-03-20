import typing as typ
from dataclasses import dataclass

import pydantic as pyd

from ..bl import PresetName, SocketName, BLEnumID
from .managed_obj import ManagedObj


class ManagedObjDef(pyd.BaseModel):
	mk: typ.Callable[[str], ManagedObj]
	name_prefix: str = ''
