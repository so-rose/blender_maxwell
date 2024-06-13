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

import functools
import weakref


def method_lru(maxsize=2048, typed=False):
	"""LRU for methods.

	Uses a weak reference to `self` to support memoized methods without memory leaks.
	"""

	def wrapped_method(method):
		@functools.lru_cache(maxsize, typed)
		def _method(_self, *args, **kwargs):
			return method(_self(), *args, **kwargs)

		@functools.wraps(method)
		def inner_method(self, *args, **kwargs):
			return _method(weakref.ref(self), *args, **kwargs)

		return inner_method

	return wrapped_method
