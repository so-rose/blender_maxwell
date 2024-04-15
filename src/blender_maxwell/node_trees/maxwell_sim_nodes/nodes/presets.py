import typing as typ

import pydantic as pyd

from .. import contracts as ct


class PresetDef(pyd.BaseModel):
	label: ct.PresetName
	description: str
	values: dict[ct.SocketName, typ.Any]
