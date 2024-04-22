import enum

from blender_maxwell.utils import blender_type_enum


class ManagedObjType(blender_type_enum.BlenderTypeEnum):
	ManagedBLImage = enum.auto()

	ManagedBLCollection = enum.auto()
	ManagedBLEmpty = enum.auto()
	ManagedBLMesh = enum.auto()
	ManagedBLVolume = enum.auto()
	ManagedBLModifier = enum.auto()
