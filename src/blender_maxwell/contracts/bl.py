# blender_maxwell
# Copyright (C) 2024 blender_maxwell Project Contributors
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import typing as typ

import bpy

####################
# - Blender Strings
####################
BLEnumID: typ.TypeAlias = str
SocketName: typ.TypeAlias = str
PropName: typ.TypeAlias = str

####################
# - Blender Enums
####################
BLImportMethod: typ.TypeAlias = typ.Literal['append', 'link']
BLModifierType: typ.TypeAlias = typ.Literal['NODES', 'ARRAY']
BLNodeTreeInterfaceID: typ.TypeAlias = str

BLIcon: typ.TypeAlias = str
BLIconSet: frozenset[BLIcon] = frozenset(
	bpy.types.UILayout.bl_rna.functions['prop'].parameters['icon'].enum_items.keys()
)

BLEnumElement = tuple[BLEnumID, str, str, BLIcon, int]

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
BLIDStruct: typ.TypeAlias = (
	bpy.types.Action
	| bpy.types.Armature
	| bpy.types.Brush
	| bpy.types.CacheFile
	| bpy.types.Camera
	| bpy.types.Collection
	| bpy.types.Curve
	| bpy.types.Curves
	| bpy.types.FreestyleLineStyle
	| bpy.types.GreasePencil
	| bpy.types.Image
	| bpy.types.Key
	| bpy.types.Lattice
	| bpy.types.Library
	| bpy.types.Light
	| bpy.types.LightProbe
	| bpy.types.Mask
	| bpy.types.Material
	| bpy.types.Mesh
	| bpy.types.MetaBall
	| bpy.types.MovieClip
	| bpy.types.NodeTree
	| bpy.types.Object
	| bpy.types.PaintCurve
	| bpy.types.Palette
	| bpy.types.ParticleSettings
	| bpy.types.PointCloud
	| bpy.types.Scene
	| bpy.types.Screen
	| bpy.types.Sound
	| bpy.types.Speaker
	| bpy.types.Text
	| bpy.types.Texture
	| bpy.types.VectorFont
	| bpy.types.Volume
	| bpy.types.WindowManager
	| bpy.types.WorkSpace
	| bpy.types.World
)
BLKeymapItem: typ.TypeAlias = typ.Any  ## TODO: Better Type
BLPropFlag: typ.TypeAlias = typ.Literal[
	'HIDDEN',
	'SKIP_SAVE',
	'SKIP_PRESET',
	'ANIMATABLE',
	'LIBRARY_EDITABLE',
	'PROPORTIONAL',
	'TEXTEDIT_UPDATE',
	'OUTPUT_PATH',
]
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
KeymapItemDef: typ.TypeAlias = typ.Any
ManagedObjName = str

####################
# - Blender Strings
####################
PresetName = str
