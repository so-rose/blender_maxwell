import enum

from blender_maxwell.utils.blender_type_enum import BlenderTypeEnum


class ManagedObjType(BlenderTypeEnum):
	ManagedBLImage = enum.auto()

	ManagedBLCollection = enum.auto()
	ManagedBLEmpty = enum.auto()
	ManagedBLMesh = enum.auto()
	ManagedBLVolume = enum.auto()
	ManagedBLModifier = enum.auto()
