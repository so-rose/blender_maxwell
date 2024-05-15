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
import uuid


class Signal(enum.StrEnum):
	"""A value used to signal the descriptor via its `__set__`.

	Such a signal **must** be entirely unique: Even a well-thought-out string could conceivably produce a very nasty bug, where instead of setting a descriptor-managed attribute, the user would inadvertently signal the descriptor.

	To make it effectively impossible to confuse any other object whatsoever with a signal, the enum values are set to per-session `uuid.uuid4()`.

	Notes:
		**Do not** use this enum for anything other than directly signalling a `bl_cache` descriptor via its setter.

		**Do not** store this enum `Signal` in a variable or method binding that survives longer than the session.

		**Do not** persist this enum; the values will change whenever `bl_cache` is (re)loaded.

	Attributes:
		CacheNotReady: The cache isn't yet ready to be used.
			Generally, this is because the `BLInstance` isn't made yet.
		CacheEmpty: The cache has no information to offer.

		InvalidateCache: The cache should be invalidated.
		InvalidateCacheNoUpdate: The cache should be invalidated, but no update method should be run.
		DoUpdate: Any update method that the cache triggers on change should be run.
			An update is **not guaranteeed** to be run, merely requested.

		ResetEnumItems: Cached dynamic enum items should be recomputed on next use.
		ResetStrSearch: Cached string-search items should be recomputed on next use.
	"""

	# Cache Management
	CacheNotReady: str = str(uuid.uuid4())
	CacheEmpty: str = str(uuid.uuid4())

	# Invalidation
	InvalidateCache: str = str(uuid.uuid4())
	InvalidateCacheNoUpdate: str = str(uuid.uuid4())
	DoUpdate: str = str(uuid.uuid4())

	# Reset Signals
	## -> Invalidates data adjascent to fields.
	ResetEnumItems: str = str(uuid.uuid4())
	ResetStrSearch: str = str(uuid.uuid4())
