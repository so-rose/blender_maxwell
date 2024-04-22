import enum

from blender_maxwell.utils import blender_type_enum


@blender_type_enum.append_cls_name_to_values
class TreeType(blender_type_enum.BlenderTypeEnum):
	MaxwellSim = enum.auto()
