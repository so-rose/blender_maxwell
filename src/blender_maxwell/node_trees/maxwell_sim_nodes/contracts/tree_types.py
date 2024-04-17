import enum

from blender_maxwell.utils.blender_type_enum import (
	BlenderTypeEnum,
	append_cls_name_to_values,
)


@append_cls_name_to_values
class TreeType(BlenderTypeEnum):
	MaxwellSim = enum.auto()
