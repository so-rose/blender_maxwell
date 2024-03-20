import enum

from ....utils.blender_type_enum import BlenderTypeEnum


class ManagedObjType(BlenderTypeEnum):
	ManagedBLObject = enum.auto()
	ManagedBLImage = enum.auto()
