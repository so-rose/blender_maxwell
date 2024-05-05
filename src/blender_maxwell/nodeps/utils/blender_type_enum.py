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

import enum


####################
# - StrEnum
####################
def prefix_values_with(prefix: str) -> type[enum.Enum]:
	"""`StrEnum` class decorator that prepends `prefix` to all class member values.

	Parameters:
		name: The name to prepend behind all `StrEnum` member values.

	Returns:
		A new StrEnum class with altered member values.
	"""
	## DO NOT USE FOR ENUMS WITH METHODS

	def _decorator(cls: enum.StrEnum):
		new_members = {
			member_name: prefix + member_value
			for member_name, member_value in cls.__members__.items()
		}

		new_cls = enum.StrEnum(cls.__name__, new_members)
		new_cls.__doc__ = cls.__doc__
		new_cls.__module__ = cls.__module__

		return new_cls

	return _decorator


####################
# - BlenderTypeEnum
####################
class BlenderTypeEnum(str, enum.Enum):
	"""Homegrown `str` enum for Blender types."""

	def _generate_next_value_(name, *_):
		return name


def append_cls_name_to_values(cls) -> type[enum.Enum]:
	"""Enum class decorator that appends the class name to all values."""
	# Construct Set w/Modified Member Names
	new_members = {
		name: f'{name}{cls.__name__}' for name, member in cls.__members__.items()
	}

	# Dynamically Declare New Enum Class w/Modified Members
	new_cls = enum.Enum(cls.__name__, new_members, type=BlenderTypeEnum)
	new_cls.__doc__ = cls.__doc__
	new_cls.__module__ = cls.__module__

	# Return New (Replacing) Enum Class
	return new_cls


def wrap_values_in_MT(cls) -> type[enum.Enum]:
	"""Enum class decorator that prepends "BLENDER_MAXWELL_MT_" to all values."""
	# Construct Set w/Modified Member Names
	new_members = {
		name: f'BLENDER_MAXWELL_MT_{name}' for name, member in cls.__members__.items()
	}

	# Dynamically Declare New Enum Class w/Modified Members
	new_cls = enum.Enum(cls.__name__, new_members, type=BlenderTypeEnum)
	new_cls.__doc__ = cls.__doc__
	new_cls.__module__ = cls.__module__
	new_cls.get_tree = cls.get_tree

	# Return New (Replacing) Enum Class
	return new_cls
