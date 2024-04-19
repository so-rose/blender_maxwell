"""Defines Operator Types as an enum, making it easy for any part of the addon to refer to any operator."""

import enum

from ..nodeps.utils import blender_type_enum
from .addon import NAME as ADDON_NAME


@blender_type_enum.prefix_values_with(f'{ADDON_NAME}.')
class OperatorType(enum.StrEnum):
	"""Identifiers for addon-defined `bpy.types.Operator`."""

	InstallPyDeps = enum.auto()
	UninstallPyDeps = enum.auto()
	ManagePyDeps = enum.auto()
