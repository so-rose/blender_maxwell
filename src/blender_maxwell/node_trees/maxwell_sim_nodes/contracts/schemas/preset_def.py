import typing as typ

import pydantic as pyd

from ..bl import PresetName, SocketName, BLEnumID

class PresetDef(pyd.BaseModel):
	label: PresetName
	description: str
	values: dict[SocketName, typ.Any]
