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


class staticproperty(property):  # noqa: N801
	"""A read-only variant of `@property` that is entirely static, for use in specific situations.

	The decorated method must take no arguments whatsoever, including `self`/`cls`.

	Examples:
		Use as usual:
		```python
		class Spam:
			@staticproperty
			def eggs():
				return 10

		assert Spam.eggs == 10
		```
	"""

	def __get__(self, *_):
		"""Overridden getter that ignores instance and owner, and just returns the value of the evaluated (static) method.

		Returns:
			The evaluated value of the static method that was decorated.
		"""
		return self.fget()
