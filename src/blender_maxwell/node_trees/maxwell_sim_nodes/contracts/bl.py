import enum
import pydantic as pyd
import typing_extensions as pytypes_ext

####################
# - Pure BL Types
####################
BLEnumID = pytypes_ext.Annotated[
	str,
	pyd.StringConstraints(
		pattern=r'^[A-Z_]+$',
	),
]
SocketName = pytypes_ext.Annotated[
	str,
	pyd.StringConstraints(
		pattern=r'^[a-zA-Z0-9_]+$',
	),
]
PresetName = pytypes_ext.Annotated[
	str,
	pyd.StringConstraints(
		pattern=r'^[a-zA-Z0-9_]+$',
	),
]
BLColorRGBA = tuple[float, float, float, float]

####################
# - Shared-With-BL Types
####################
ManagedObjName = pytypes_ext.Annotated[
	str,
	pyd.StringConstraints(
		pattern=r'^[a-z_]+$',
	),
]
