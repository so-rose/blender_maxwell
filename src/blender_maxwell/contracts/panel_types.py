"""Defines Panel Types as an enum, making it easy for any part of the addon to refer to any panel."""

import enum

from blender_maxwell.nodeps.utils import blender_type_enum

from .addon import NAME as ADDON_NAME


@blender_type_enum.prefix_values_with(f'{ADDON_NAME.upper()}_PT_')
class PanelType(enum.StrEnum):
	"""Identifiers for addon-defined `bpy.types.Panel`."""
