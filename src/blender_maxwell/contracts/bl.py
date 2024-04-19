import typing as typ

import bpy

####################
# - Blender Strings
####################
BLEnumID = str
SocketName = str

####################
# - Blender Enums
####################
BLImportMethod: typ.TypeAlias = typ.Literal['append', 'link']
BLModifierType: typ.TypeAlias = typ.Literal['NODES', 'ARRAY']
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
BLSpaceType: typ.TypeAlias = typ.Literal[
	'EMPTY',
	'VIEW_3D',
	'IMAGE_EDITOR',
	'NODE_EDITOR',
	'SEQUENCE_EDITOR',
	'CLIP_EDITOR',
	'DOPESHEET_EDITOR',
	'GRAPH_EDITOR',
	'NLA_EDITOR',
	'TEXT_EDITOR',
	'CONSOLE',
	'INFO',
	'TOPBAR',
	'STATUSBAR',
	'OUTLINER',
	'PROPERTIES',
	'FILE_BROWSER',
	'SPREADSHEET',
	'PREFERENCES',
]
BLRegionType: typ.TypeAlias = typ.Literal[
	'WINDOW',
	'HEADER',
	'CHANNELS',
	'TEMPORARY',
	'UI',
	'TOOLS',
	'TOOL_PROPS',
	'ASSET_SHELF',
	'ASSET_SHELF_HEADER',
	'PREVIEW',
	'HUD',
	'NAVIGATION_BAR',
	'EXECUTE',
	'FOOTER',
	'TOOL_HEADER',
	'XR',
]
BLOperatorStatus: typ.TypeAlias = set[
	typ.Literal['RUNNING_MODAL', 'CANCELLED', 'FINISHED', 'PASS_THROUGH', 'INTERFACE']
]

####################
# - Addon Types
####################
KeymapItemDef: typ.TypeAlias = typ.Any  ## TODO: Better Type
ManagedObjName = str

####################
# - Blender Strings
####################
PresetName = str
