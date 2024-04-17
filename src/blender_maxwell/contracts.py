import typing as typ

import bpy
import pydantic as pyd
import typing_extensions as typx

####################
# - Blender Strings
####################
BLEnumID = typx.Annotated[
	str,
	pyd.StringConstraints(
		pattern=r'^[A-Z_]+$',
	),
]
SocketName = typx.Annotated[
	str,
	pyd.StringConstraints(
		pattern=r'^[a-zA-Z0-9_]+$',
	),
]

####################
# - Blender Enums
####################
BLModifierType: typ.TypeAlias = typx.Literal['NODES', 'ARRAY']
BLNodeTreeInterfaceID: typ.TypeAlias = str

BLIconSet: frozenset[str] = frozenset(
	bpy.types.UILayout.bl_rna.functions['prop'].parameters['icon'].enum_items.keys()
)


####################
# - Blender Structs
####################
BLClass: typ.TypeAlias = (
	bpy.types.Panel
	| bpy.types.UIList
	| bpy.types.Menu
	| bpy.types.Header
	| bpy.types.Operator
	| bpy.types.KeyingSetInfo
	| bpy.types.RenderEngine
	| bpy.types.AssetShelf
	| bpy.types.FileHandler
)
BLKeymapItem: typ.TypeAlias = typ.Any  ## TODO: Better Type
BLColorRGBA = tuple[float, float, float, float]

####################
# - Operators
####################
BLOperatorStatus: typ.TypeAlias = set[
	typx.Literal['RUNNING_MODAL', 'CANCELLED', 'FINISHED', 'PASS_THROUGH', 'INTERFACE']
]

####################
# - Addon Types
####################
ManagedObjName = typx.Annotated[
	str,
	pyd.StringConstraints(
		pattern=r'^[a-z_]+$',
	),
]
KeymapItemDef: typ.TypeAlias = typ.Any  ## TODO: Better Type

####################
# - Blender Strings
####################
PresetName = typx.Annotated[
	str,
	pyd.StringConstraints(
		pattern=r'^[a-zA-Z0-9_]+$',
	),
]
